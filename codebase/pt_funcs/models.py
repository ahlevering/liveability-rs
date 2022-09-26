import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50

from codebase.pt_funcs.run_tracker import VarTrackerLBM

class LBMBaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.model(x)

class DimensionMLP(nn.Module):
    def __init__(self, ndims):
        super().__init__()
        self.reduce = nn.Linear(2048, ndims)
        self.dim_score = nn.Linear(ndims, 1)
    
    def forward(self, x):
        x = self.reduce(x)
        features = x.relu()
        score = self.dim_score(features)
        return score

class LBMDimensionNet(nn.Module):
    def __init__(self, score_info):
        super().__init__()
        self.score_info = score_info
        
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        
        # Initialize 1 MLP for each dimension
        self.dim_score_mlps = nn.ModuleDict()
        for dim in score_info['dim_scores']:
            self.dim_score_mlps[dim] = DimensionMLP(250)
        self.lbm_score_layer = nn.Linear(len(score_info['dim_scores']), 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Calculate and extract all features
        dim_scores = torch.stack([self.dim_score_mlps[dim](features) for dim in self.score_info['dim_scores']])
        dim_scores = dim_scores.squeeze().transpose(1,0)
        lbm_score = self.lbm_score_layer(dim_scores)
        return dim_scores, lbm_score        

class LBMDimensionNetFeatures(nn.Module):
    def __init__(self, score_info):
        super().__init__()
        self.score_info = score_info
        
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        
        # Initialize 1 MLP for each dimension
        self.dim_score_mlps = nn.ModuleDict()
        for dim in score_info['dim_scores']:
            self.dim_score_mlps[dim] = DimensionMLP(250)
        self.lbm_score_layer = nn.Linear(len(score_info['dim_scores']), 1)

    def forward(self, x):
        features = self.feature_extractor(x)        
        
        # Calculate and extract all features
        # dim_features, dim_scores = torch.stack([self.dim_score_mlps[dim](features) for dim in self.score_info['dim_scores']])
        dim_outputs = [self.dim_score_mlps[dim](features) for dim in self.score_info['dim_scores']]
        dim_features = [d[0] for d in dim_outputs]
        dim_scores = [d[1] for d in dim_outputs]
        dim_scores = torch.stack(dim_scores)
        
        # dim_scores = torch.stack([self.dim_score_mlps[dim](features) for dim in self.score_info['dim_scores']])
        dim_scores = dim_scores.squeeze(1).transpose(1,0)
        lbm_score = self.lbm_score_layer(dim_scores)
        return features, dim_features, dim_scores, lbm_score

class LBMDimensionModel(pl.LightningModule):
    def __init__(self, scatter_dir, run_name, label_info, splits):
        super().__init__()
        self.lr = None
        self.decay = None
        self.model = LBMDimensionNet(label_info)

        self.score_info = label_info
        self.run_name = run_name
        self.out_dir = scatter_dir

        if 'train' in splits:
            self.train_tracker = VarTrackerLBM(self.out_dir, 'train', label_info)
        if 'val' in splits:
            self.val_tracker = VarTrackerLBM(self.out_dir, 'val', label_info)
        if 'test' in splits or 'all' in splits:
            self.test_tracker = VarTrackerLBM(self.out_dir, 'test', label_info)

    def forward(self, x):
        x = self.model(x)
        return x

    # def on_train_start(self):
    #     # Hotfix for lightning not casting sub-modules to device
    #     device = next(self.model.parameters()).device
    #     for dim in self.model.dim_score_mlps:
    #         self.model.dim_score_mlps[dim].to(device)
    
    # def on_test_start(self):
    #     # Hotfix for lightning not casting sub-modules to device
    #     device = next(self.model.parameters()).device
    #     for dim in self.model.dim_score_mlps:
    #         self.model.dim_score_mlps[dim].to(device)        

    def iteration_forward(self, batch, tracker, split):
        ### Model forward ###
        dim_scores, lbm_score = self.model(batch['patches'])

        ### Fix some issues with batch variables ###
        lbm_score = lbm_score.squeeze().double()
        dim_scores = dim_scores.double()
        lbm_labels = batch['lbm_score'].double()
        dim_labels = torch.stack(batch['dim_scores']).transpose(1,0).double()

        ### Add variables to trackers ###
        ids = batch['ids'].cpu().numpy()
        lat = batch['lat'].cpu().numpy()
        lon = batch['lon'].cpu().numpy()

        ## Dim scores
        if 'dim_scores' in self.score_info:
            for index, dim in enumerate(self.score_info['dim_scores']):
                dim_preds = dim_scores[:,index].detach().cpu().numpy()
                dim_gt = dim_labels[:,index].cpu().numpy()              
                dim_datapts = {'ids':ids, 'lat':lat, 'lon':lon, 'preds':dim_preds, 'gt':dim_gt}
                self.datapoints_to_tracker(tracker, dim_datapts, dim)

        # LBM score
        lbm_preds = lbm_score.detach().cpu().numpy()
        lbm_gt = batch['lbm_score'].cpu().numpy()              
        lbm_datapts = {'ids':ids, 'lat':lat, 'lon':lon, 'preds':lbm_preds, 'gt':lbm_gt}
        self.datapoints_to_tracker(tracker, lbm_datapts, 'lbm_score')

        ### Calculate overall loss ###
        subscore_loss, score_loss = self.get_bottleneck_loss(dim_scores, lbm_score, dim_labels, lbm_labels)           
        combined_loss = self.combine_losses(subscore_loss, score_loss)

        # Log score
        self.log(f'{split}_combined_mse', combined_loss.detach().cpu(), on_step=False, on_epoch=True)        
        return combined_loss   

    def get_bottleneck_loss(self, dim_scores, lbm_score, dim_labels, lbm_score_label):
        subscore_losses = F.mse_loss(dim_scores, dim_labels)#, reduction='none').mean()
        score_loss = F.mse_loss(lbm_score, lbm_score_label)#, reduction='none').mean()
        return subscore_losses, score_loss

    def combine_losses(self, subscore_loss, score_loss):
        return (subscore_loss * 100) + score_loss

    def datapoints_to_tracker(self, tracker, datapts, var_name):
        # Check for batch size
        length_is_one = len(datapts['preds']) == 1
        for attr in datapts:
            if length_is_one:
                datapts[attr] = [int(datapts[attr])] 
            else:
                datapts[attr] = datapts[attr].squeeze()
            
            # Store into tracker
            tracker.variables[var_name].attrs[attr].extend(datapts[attr])

    def end_epoch(self, tracker):
        tracker.store_epoch_metrics()
        
        ## Write outputs
        tracker.save_metrics_to_file()
        tracker.save_observations_to_file(self.current_epoch)
        tracker.save_scatterplot(self.current_epoch)

        ## Reset for next epoch
        tracker.reset_epoch_vars()
        # tracker.print_results()       

### Training ###
    def training_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.train_tracker, 'train')
        return loss

    def training_epoch_end(self, train_outputs):
        self.end_epoch(self.train_tracker)

### Validation ###
    def validation_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.val_tracker, 'val')
        return loss

    def validation_epoch_end(self, train_outputs):
        self.end_epoch(self.val_tracker)

### Testing ###
    def test_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.test_tracker, 'test')
        return loss

    def test_epoch_end(self, test_outputs):
        self.end_epoch(self.test_tracker)

### Optimization ###
    def set_hyperparams(self, lr=0.0001, decay=0.0001):
        self.lr = lr
        self.decay = decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)

        return [optimizer], [scheduler]        

class LBMBaselineModel(LBMDimensionModel):
    def __init__(self, scatter_dir, run_name, score_info, splits):
        super().__init__(scatter_dir, run_name, score_info, splits)
        self.model = LBMBaselineNet()

### General iteration functions ###
    def forward(self, x):
        x = self.model(x)
        return x

    def iteration_forward(self, batch, tracker, split):
        ### Model forward ###
        lbm_score = self.model(batch['patches'])

        ### Fix some issues with batch variables ###
        lbm_score = lbm_score.squeeze().double()
        lbm_labels = batch['lbm_score'].double()

        ### Add variables to trackers ###
        ids = batch['ids'].cpu().numpy()
        lat = batch['lat'].cpu().numpy()
        lon = batch['lon'].cpu().numpy()

        # LBM score
        lbm_preds = lbm_score.detach().cpu().numpy()
        lbm_gt = batch['lbm_score'].cpu().numpy()              
        lbm_datapts = {'ids':ids, 'lat':lat, 'lon':lon, 'preds':lbm_preds, 'gt':lbm_gt}
        self.datapoints_to_tracker(tracker, lbm_datapts, 'lbm_score')

        ### Calculate overall loss ###      
        score_loss = F.mse_loss(lbm_score, lbm_labels)

        # Log score
        self.log(f'{split}_lbm_mse', score_loss.detach().cpu(), on_step=False, on_epoch=True)        
        return score_loss