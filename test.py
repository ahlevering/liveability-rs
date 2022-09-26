import yaml
import warnings

from datetime import datetime
from sklearn.manifold import TSNE

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch

from codebase.pt_funcs.dataloaders import LBMData, LBMLoader
from codebase.pt_funcs.models import LBMDimensionModel, LBMBaselineModel

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer

##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)

# Shut up please, GeoPandas
# warnings.simplefilter(action='ignore', category=UserWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/full_model.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']
train_dir = f"{exp_params['paths']['train_dir']}"

##### SET UP AUGMENTATIONS #####
test_trans = process_yaml.setup_transforms(exp_params['transforms']['test'])

label_info = {}
label_info['dim_scores'] = {}
if 'labels' in exp_params:
    if 'dim_scores' in exp_params['labels']:
        if type(exp_params['labels']['dim_scores']) == list:
            for i, key in enumerate(exp_params['labels']['dim_scores']):
                label_info['dim_scores'][key] = {}
                # label_info['dim_scores'][key]['index'] = i
                label_info['dim_scores'][key]['ylims'] = [-1.5, 1.5]

label_info['lbm_score'] = {}
# label_info['lbm_score']['index'] = 0
label_info['lbm_score']['ylims'] = [-1.5, 1.5]

##### SETUP LOADERS #####
lbm_module = LBMLoader( exp_params['hyperparams']['workers'],    
                        exp_params['hyperparams']['batch_size'],
                        data_class=LBMData
                    )

lbm_module.setup_data_classes(  exp_params['paths']['splits_file'],
                                exp_params['paths']['images_root'],
                                label_info['dim_scores'],
                                exp_params['descriptions']['splits'],
                                test_transforms=test_trans
                            )

loader_emulator = next(iter(lbm_module.test_data))

##### STORE ENVIRONMENT AND FILES #####
run_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(f"\nExperiment start: {run_time}\n")
base_path = train_dir+f'test/{run_time}/' # Add to existing dir
organizer = ExperimentOrganizer(base_path)
organizer.store_yaml(setup_file)
organizer.store_environment()
organizer.store_codebase(['.py'])

##### SETUP MODEL #####
### Model loading depends on state dicts specified in the set-up yaml ###
## Loading a fully-trained model

if 'labels' in exp_params:
    model = LBMDimensionModel(organizer.root_path+'outputs/', run_name, label_info, exp_params['descriptions']['splits'])
else:
    model = LBMBaselineModel(organizer.root_path+'outputs/', run_name, label_info, exp_params['descriptions']['splits'])

if 'weights_file' in exp_params['paths']:
    weights_file = train_dir + exp_params['paths']['weights_file']
    state_dict = torch.load(weights_file, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)

model = model.eval()

##### SETUP TRAINER #####
tb_logger = TensorBoardLogger(save_dir=organizer.logs_path)
tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
##### FIT MODEL #####
print(f"testing {run_name}")
tester.test(model, datamodule=lbm_module)

# outputs = tester.model.batches_with_vector

# out_dict = {}
# for key in outputs[0]:
#     out_dict[key] = []
# for batch in outputs:
#     for key in batch:
#         try:
#             out_dict[key].extend(batch[key])
#         except:
#             out_dict[key].append(batch[key])

# subscore_names = [x for x in out_dict.keys() if 'highdim' in x or x == 'merged_features']

# def scale_to_01_range(x):
#     value_range = (np.max(x) - np.min(x))
#     starts_from_zero = x - np.min(x)
#     return starts_from_zero / value_range

# ## Calculate principal components

# # pca_arrays = []
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# for subscore_name in subscore_names:
#     ss_vec = out_dict[subscore_name]
#     ss_name = subscore_name.split('_')[-1]
    
#     ## Run PCA
#     pca = PCA(3)
#     pca_arrays = pca.fit(ss_vec)
#     pca_vectors = pca_arrays.fit_transform(ss_vec)
#     for dim in range(pca_vectors.shape[1]):
#         pc = pca_vectors[:, dim].squeeze()
#         pc = (pc - np.min(pc)) / np.ptp(pc) * 255

#         out_dict[f'{ss_name}_pca_{dim}'] = pc.tolist()

#     ## PCA reduction for big vector
#     if ss_name == 'merged_features':
#         ss_vec = pca_vectors

#     ## Run TSNE
#     tsne = TSNE(perplexity=300, learning_rate=500, n_iter=1000, early_exaggeration=150, random_state=113)
#     embedding = tsne.fit_transform(ss_vec)

#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     out_dict[f'{ss_name}_tsne_x'] = scale_to_01_range(embedding[:, 0])
#     out_dict[f'{ss_name}_tsne_y'] = scale_to_01_range(embedding[:, 1])

#     # ax.scatter(out_dict[f'{ss_name}_tsne_x'], out_dict[f'{ss_name}_tsne_y'], s=0.1)
#     # plt.savefig("test2.jpg")

#     del out_dict[subscore_name]

# # import pandas as pd
# import geopandas as gpd
# out_df = gpd.GeoDataFrame(out_dict, crs=28992)

# for col in out_df:
#     try:
#         out_df[col] = out_df[col].astype(float)
#     except:
#         pass

# from shapely.geometry import Point
# def fix_polygon(row):
#     centroid = row['geom'].centroid
#     grid_true_center_x = centroid.xy[0][0] - (centroid.xy[0][0] % 100) + 50
#     grid_true_center_y = centroid.xy[1][0] - (centroid.xy[1][0] % 100) + 50
#     return Point([grid_true_center_x, grid_true_center_y]).buffer(50, cap_style = 3)

# # out_df['geometry'] = out_df['geom']
# out_df['geometry'] = out_df.apply(fix_polygon, axis=1)
# out_df = out_df.drop(['geom'], axis=1)
# out_df.set_geometry('geometry', inplace=True)
# out_df.to_file(f"data/outputs_geo/{run_name}.geojson", driver='GeoJSON')