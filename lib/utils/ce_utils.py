import math

import torch
import torch.nn.functional as F


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape
    for i in range(b):
        bbox_i = bbox[i].cpu().tolist()
        bbox_mask[i, int(bbox_i[1]):int(bbox_i[1] + bbox_i[3] - 1), int(bbox_i[0]):int(bbox_i[0] + bbox_i[2] - 1)] = 1
    return bbox_mask

def generate_policy(cfg, bs, device,type):
    stride = cfg.MODEL.BACKBONE.STRIDE
    if (type == "dynamic"):
        dynamic_size = cfg.DATA.DYNAMIC.SIZE
        dynamic_feat_size = dynamic_size // stride
        if dynamic_feat_size == 12:
            index = slice(2,10)
        elif dynamic_feat_size == 10:
            index = slice(1, 9)
        else:
            raise NotImplementedError
        box_mask_d = torch.zeros([bs, dynamic_feat_size, dynamic_feat_size], device=device)
        box_mask_d[:, index, index] = 1
        box_mask_d = box_mask_d.flatten(1)
        dynamic_policy = torch.zeros([bs, dynamic_feat_size * dynamic_feat_size, 5], device=device)
        dynamic_policy[:, :, 1] = box_mask_d
        background = box_mask_d
        background = torch.where(background == 1, 5, background)
        background = torch.where(background == 0, 1, background)
        background = torch.where(background == 5, 0, background)
        dynamic_policy[:, :, 2] = background
        return dynamic_policy
    if (type == "search"):
        search_size = cfg.DATA.SEARCH.SIZE
        search_feat_size = search_size // stride
        search_policy = torch.zeros([bs, search_feat_size * search_feat_size, 5], device=device)
        search_policy[:, :, 4] = 1
        return search_policy
    if (type == "template"):
        template_size = cfg.DATA.TEMPLATE.SIZE
        template_feat_size = template_size // stride
        template_policy = torch.zeros([bs, template_feat_size * template_feat_size, 5], device=device)
        template_policy[:, :, 0] = 1
        return template_policy

def generate_mask_cond(cfg, bs, device, gt_bbox,type):
    if (type == "template"):
        template_size = cfg.DATA.TEMPLATE.SIZE
        stride = cfg.MODEL.BACKBONE.STRIDE
        template_feat_size = template_size // stride
    elif (type == "dynamic"):
        template_size = cfg.DATA.DYNAMIC.SIZE
        stride = cfg.MODEL.BACKBONE.STRIDE
        template_feat_size = template_size // stride

    if cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'ALL':
        box_mask_z = None
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT':
        if template_feat_size == 8:
            index = slice(3, 4)
        elif template_feat_size == 12:
            index = slice(5, 6)
        elif template_feat_size == 10:
            index = slice(4, 5)
        elif template_feat_size == 7:
            index = slice(3, 4)
        elif template_feat_size == 14:
            index = slice(6, 7)
        elif template_feat_size == 6:
            index = slice(2, 3)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_REC':
        # use fixed 4x4 region, 3:5 for 8x8
        # use fixed 4x4 region 5:6 for 12x12
        if template_feat_size == 8:
            index = slice(3, 5)
        elif template_feat_size == 12:
            index = slice(5, 7)
        elif template_feat_size == 7:
            index = slice(3, 4)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'GT_BOX':
        box_mask_z = torch.zeros([bs, template_size, template_size], device=device)
        # box_mask_z_ori = data['template_seg'][0].view(-1, 1, *data['template_seg'].shape[2:])  # (batch, 1, 128, 128)
        box_mask_z = generate_bbox_mask(box_mask_z, gt_bbox * template_size).unsqueeze(1).to(
            torch.float)  # (batch, 1, 128, 128)
        # box_mask_z_vis = box_mask_z.cpu().numpy()
        box_mask_z = F.interpolate(box_mask_z, scale_factor=1. / cfg.MODEL.BACKBONE.STRIDE, mode='bilinear',
                                   align_corners=False)
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
        # box_mask_z_vis = box_mask_z[:, 0, ...].cpu().numpy()
        # gaussian_maps_vis = generate_heatmap(data['template_anno'], self.cfg.DATA.TEMPLATE.SIZE, self.cfg.MODEL.STRIDE)[0].cpu().numpy()
    else:
        raise NotImplementedError

    return box_mask_z


def adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1, iters=-1):
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    if iters == -1:
        iters = epoch * ITERS_PER_EPOCH
    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)
    iters = iters - ITERS_PER_EPOCH * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate
