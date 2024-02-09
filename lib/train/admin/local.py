class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/janani/Research/Trackers/Git_hub_Submision/oiftrack/pretrained_networks'
        self.lasot_dir = '/media/disk1/dataset/LaSOT'
        self.got10k_dir = '/media/disk1/dataset/GOT-10K/train'
        self.got10k_val_dir = '/media/disk1/dataset/GOT-10K/val'
        self.lasot_lmdb_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/media/disk1/dataset/TrackingNet'
        self.trackingnet_lmdb_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/trackingnet_lmdb'
        self.coco_dir = '/media/disk1/dataset/Coco'
        self.coco_lmdb_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
