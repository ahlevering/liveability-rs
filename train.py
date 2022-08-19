import yaml
from datetime import datetime

import warnings

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
import torch

from codebase.pt_funcs.dataloaders import LBMData, LBMLoader
from codebase.pt_funcs.models import LBMDimensionModel, LBMBaselineModel

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer

##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)

# Shut up please, GeoPandas
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/baseline.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

##### SET UP TRANSFORMS #####
train_trans = process_yaml.setup_transforms(exp_params['transforms']['train'])

val_trans = process_yaml.setup_transforms(exp_params['transforms']['val'])

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
                                train_transforms=train_trans,
                                val_transforms=val_trans
                            )

loader_emulator = next(iter(lbm_module.train_data))

##### STORE ENVIRONMENT AND FILES #####
run_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(f"\nExperiment start: {run_time}\n")
base_path = f"runs/{run_family}/{run_name}/{run_time}/"
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
model.set_hyperparams(lr=exp_params['hyperparams']['lr'], decay=exp_params['hyperparams']['decay'])
if 'weights_file' in exp_params['paths']:
    state_dict = torch.load(exp_params['paths']['weights_file'], map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)

##### SETUP TRAINER #####
if 'labels' in exp_params:
    if 'dim_scores' in exp_params['labels']:
        to_monitor = 'val_combined_mse'
        filename = '{epoch:02d}-{val_combined_mse:.2f}'
else:
    to_monitor = 'val_lbm_mse'
    filename = '{epoch:02d}-{val_lbm_mse:.2f}'

checkpoint_callback = ModelCheckpoint(
    monitor= to_monitor,
    dirpath= str(organizer.states_path),
    filename=filename,
    save_top_k=999,
    mode='min')

tb_logger = TensorBoardLogger(save_dir=organizer.logs_path)
trainer = Trainer(  max_epochs=exp_params['hyperparams']['epochs'],
                    gpus=exp_params['hyperparams']['gpu_nums'],
                    check_val_every_n_epoch=exp_params['hyperparams']['check_val_every_n'],
                    callbacks=[checkpoint_callback],
                    logger = tb_logger,
                    fast_dev_run=False,
                    auto_lr_find=False,
                    # profiler="simple",
                    # limit_train_batches=10,
                    # limit_val_batches=10,
                )

##### FIT MODEL #####
print(f"fitting {run_name}")
trainer.fit(model, datamodule=lbm_module)