import math

from lib.models.oiftrack import build_oiftrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond,generate_policy


class OIFTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OIFTrack, self).__init__(params)
        network = build_oiftrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.d_dict1 = {}
        self.z_dict_list = []
        self.policy = []
        # Set the update interval
        self.update_intervals = [40]
        self.num_extra_template = self.cfg.DATA.DYNAMIC.NUMBER

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.z_dict_list.append(self.z_dict1.tensors)
        d_resize_factor = None
        for i in range(self.num_extra_template):
            d_patch_arr, d_resize_factor, d_amask_arr = sample_target(image, info['init_bbox'],
                                                                    self.params.dynamic_factor,
                                                                    output_sz=self.params.dynamic_size)
            dynamic = self.preprocessor.process(d_patch_arr, d_amask_arr)
            with torch.no_grad():
                self.d_dict1 = dynamic

            self.z_dict_list.append(self.d_dict1.tensors)

        self.box_mask_z = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device, crop_type='template').squeeze(1)
            self.box_mask_z.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox,"template"))
            for i in range(self.num_extra_template):
                dynamic_bbox = self.transform_bbox_to_crop(info['init_bbox'], d_resize_factor,
                                                            template.tensors.device, crop_type='dynamic').squeeze(1)
                self.box_mask_z.append(generate_mask_cond(self.cfg, 1, template.tensors.device, dynamic_bbox,"dynamic"))

        template_policy = generate_policy(self.cfg, self.z_dict_list[1].shape[0], self.z_dict_list[1].device, "template")
        self.policy.append(template_policy)
        dynamic_policy = generate_policy(self.cfg, self.z_dict_list[1].shape[0], self.z_dict_list[1].device, "dynamic")
        self.policy.append(dynamic_policy)
        search_policy = generate_policy(self.cfg, self.z_dict_list[1].shape[0], self.z_dict_list[1].device, "search")
        self.policy.append(search_policy)
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict_list, search=x_dict.tensors, ce_template_mask=self.box_mask_z, policy=self.policy)

        # add hann windows
        pred_score_map = out_dict['score_map']
        conf_score = out_dict['score']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.65:
                z_patch_arr2, _, z_amask_arr2 = sample_target(image, self.state, self.params.dynamic_factor,
                                                              output_sz=self.params.dynamic_size)
                template_t = self.preprocessor.process(z_patch_arr2, z_amask_arr2)
                self.z_dict_list.pop(1)
                self.z_dict_list.append(template_t.tensors)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return OIFTrack