"""GlaS official split, 10% labelled training, remaining training unlabeled."""
from copy import deepcopy
from box import Box
from configs.Glomeruli_uni2_spherical_hrrefine import config as base_config
config=deepcopy(base_config)
config['name']='glas_10pct_semi'; config['seed']=42; config['batch_size']=2; config['num_workers']=0; config['pin_memory']=False; config['persistent_workers']=False; config['prefetch_factor']=None
config['dataset'].update({'dataset_root':'Glas/GlasDataset','dataset_csv_path':'dataset_cfg/glas_10pct.csv','data_ext':'.bmp','image_folder':'','mask_folder':'','split_folders':False,'mask_suffix':'_anno','binary_foreground':True,'image_hw':(1024,1024),'num_classes':1,'train_label_column':'labeled','train_labeled_value':1,'mask_color_map':None,'mask_color_mode':'bgr'})
config['semi'].update({'enabled':True,'unlabeled_source_csv_path':'dataset_cfg/glas_10pct.csv','exclude_labeled_from_unlabeled':True,'labeled_filter_column':'labeled','labeled_filter_value':1})
config['model']['geometry']['enabled']=True; config['model']['geometry']['separator_enabled']=False
config['semi']['prototype']['enabled']=False; config['semi']['prototype']['memory']['enabled']=False
config['semi']['geometry'].update({'enabled':True,'separator_enabled':False,'unsup_coef':0.0})
cfg=Box(config)
