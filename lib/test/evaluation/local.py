from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/got10k_lmdb'
    settings.got10k_path = '/media/disk1/dataset/GOT-10K'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/janani/Research/dataset/LaSOT'
    settings.network_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/nfs'
    settings.otb_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/otb'
    settings.prj_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack'
    settings.result_plot_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/output/test/result_plots'
    settings.results_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/output'
    settings.segmentation_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/janani/Research/dataset/TrackingNet/TrackingNet'
    settings.uav_path = '/home/janani/Research/dataset/UAV123'
    settings.vot18_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/vot2018'
    settings.vot22_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/vot2022'
    settings.vot_path = '/home/janani/Research/Trackers/Git_hub_Submision/OIFTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

