from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib as mpl
from scipy.stats import kendalltau, linregress
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

mpl.use('Agg')

class VarTracker():
    def __init__(self, var_name, out_dir, split, ylims=[0,1]):
        self.var_name = var_name
        self.split = split
        self.ylims = ylims    

        ## Setup output directories
        self.metrics_dir = out_dir+'metrics/'+f'{self.split}/'
        Path(self.metrics_dir).mkdir(exist_ok=True, parents=True)
        self.observ_dir = out_dir+'observations/'+f'{self.split}/'
        Path(self.observ_dir).mkdir(exist_ok=True, parents=True)
        self.plots_dir = out_dir+'plots/'+f'{self.split}/{var_name}/'
        Path(self.plots_dir).mkdir(exist_ok=True, parents=True)        

        ## Initialize empty tracking variables
        self.attrs = {}
        for attr in ['ids', 'lat', 'lon', 'preds', 'gt']:
            self.attrs[attr] = []

        self.metrics = {}
        for metric in ['rmse', 'rsquared', 'tau_corr', 'tau_pvalues']:
            self.metrics[metric] = []
    
    def metrics_to_df(self):
        metrics_dict = {"rmse": self.metrics['rmse'],
                        "rsquared": self.metrics['rsquared'],
                        "tau_corr": self.metrics['tau_corr'],
                        "tau_pval": self.metrics['tau_pvalues']
                    }
        metrics_df = pd.DataFrame.from_dict(metrics_dict)
        return metrics_df

    def observ_to_gdf(self):
        observ_dict = { "id": self.attrs['ids'],
                        "lat": self.attrs['lat'],
                        "lon": self.attrs['lon'],
                        "pred": self.attrs['preds'],
                        "gt": self.attrs['gt'],
                        "residuals": self.attrs['gt'] - self.attrs['preds']
                    }
        observ_df = pd.DataFrame.from_dict(observ_dict)
        observ_gdf = gpd.GeoDataFrame(observ_df, geometry=gpd.points_from_xy(observ_df.lon, observ_df.lat), crs=28992)
        observ_gdf['geometry'] = observ_gdf.buffer(50, cap_style=3) # Recover square grid cells
        return observ_gdf

    def calc_epoch_metrics(self):
        ## Prevent batch sizes of 1 from messing up the metric calculations
        if self.attrs['preds'][0].size == 1:
            self.attrs['preds'] = np.array(self.attrs['preds'])
            self.attrs['gt'] = np.array(self.attrs['gt'])
        else:
            self.attrs['preds'] = np.concatenate(self.attrs['preds'])
            self.attrs['gt'] = np.concatenate(self.attrs['gt'])
        
        ## Calculate metrics
        _, _, r_value, _, _ = linregress(self.attrs['preds'], self.attrs['gt'])
        self.metrics['rsquared'].append(round(r_value, 4))
        self.metrics['rmse'].append(round(mean_squared_error(self.attrs['preds'], self.attrs['gt'], squared=False), 4))
        tau = kendalltau(self.attrs['preds'], self.attrs['gt'])
        self.metrics['tau_corr'].append(round(tau.correlation, 4))
        self.metrics['tau_pvalues'].append(round(tau.pvalue, 4))

    def save_scatterplot(self, epoch, color="#1f77b4"): # #ff7f0e
        fig = plt.figure()
        plt.scatter(self.attrs['gt'], self.attrs['preds'], c=[color])
        title = f"{self.var_name} epoch {epoch} {self.split}"
        plt.suptitle(title, fontsize=18)
        plt.xlabel("GT", fontsize=14)
        plt.ylabel("Predicted", fontsize=14)
        plt.ylim(self.ylims)
        plt.xlim(self.ylims)
        # Store plot
        plt.savefig(f"{self.plots_dir}{epoch}_{self.split}.png")
        plt.close('all')
    
    def save_metrics_to_file(self):
        metrics_df = self.metrics_to_df()
        metrics_df.to_csv(f"{self.metrics_dir}{self.var_name}.csv")

    def save_observations_to_file(self, epoch):
        observ_gdf = self.observ_to_gdf()
        observ_gdf.to_file(f"{self.observ_dir}{epoch}_{self.var_name}.geojson", driver="GeoJSON")        

class VarTrackerLBM():
    def __init__(self, out_dir, split, label_info):
        self.label_info = label_info
        self.split = split

        self.plot_colors = {'train':"#1f77b4", 'val':"#ff6600", 'test':"#4b8b3b"}
        self.variables = {}
        for key in self.label_info['dim_scores']:
            self.variables[key] = VarTracker(key, out_dir, split, label_info['dim_scores'][key]['ylims'])   
        self.variables['lbm_score'] = VarTracker('lbm_score', out_dir, split, self.label_info['lbm_score']['ylims'])

    def store_epoch_metrics(self):
        for key in self.variables:
            self.variables[key].calc_epoch_metrics()

    def save_scatterplot(self, epoch):
        color = self.plot_colors[self.split]
        for key in self.variables:
            self.variables[key].save_scatterplot(epoch, color)      

    def save_metrics_to_file(self):
        for key in self.variables:
            self.variables[key].save_metrics_to_file()

    def save_observations_to_file(self, epoch):
        for key in self.variables:
            self.variables[key].save_observations_to_file(epoch)            
    
    def reset_epoch_vars(self):
        for key in self.variables:
            for attr in self.variables[key].attrs:
                self.variables[key].attrs[attr] = []
    
    def print_results(self):
        for key in self.variables:
            print(f"{self.split} {key} RMSE: {self.variables[key].metrics['rmse'][-1]}")
            print(f"{self.split} {key} Tau: {self.variables[key].metrics['tau_corr'][-1]}")        
            print(f"{self.split} {key} R2: {self.variables[key].metrics['rsquared'][-1]}\n")