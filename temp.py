import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from scipy.signal import resample, argrelmax, argrelmin
from scipy import interpolate
sys.path.append('/Users/mitayukiya/.pyenv/versions/3.11.4/lib/python3.11/site-packages')
from PyEMD import EMD

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

import time

import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

def compute_imf_dde(r, T, iter_num, epsilon):
    emd = EMD(energy_ratio_thr=epsilon)
    imf = emd(r, max_imf=iter_num)
    D_list = []
    for k in range(iter_num):
        try:
            D = 0.5 * np.log(np.sum(np.abs(imf[k])**2)) + 0.5 * np.log(2*np.pi*np.e / T)
        except:
            D = 0.5 * np.log(2*np.pi*np.e / T)
        D_list.append(D)
    return D_list

def preprocessing(self, x: torch.Tensor, S = 20, T = 100, iter_num = 3, epsilon = 0.2, max_workers=3) -> torch.Tensor:
    resample_x = np.array(x)
    train_data = []

    for b in range(resample_x.shape[0]):
        M = int((resample_x.shape[-1] - T) / S)
        data_list = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(M):
                r_slice = resample_x[b, :, i*S:i*S+T]
                for j in range(r_slice.shape[0]):
                    r = r_slice[j, :]
                    futures.append(executor.submit(compute_imf_dde, r, T, iter_num, epsilon))
            
            results = [future.result() for future in futures]
        
        # Reshape the results into the required format
        DDE_array = np.array(results).reshape(resample_x.shape[1], M, iter_num)
        data_list = np.transpose(DDE_array, (0, 2, 1)).reshape(-1, M * iter_num).T
        
        train_data.append(data_list)

    train_data = np.array(train_data)
    train_data = torch.tensor(train_data, dtype=torch.float32)

    return train_data


# def preprocessing(x: torch.Tensor, S: int, T: int, iter_num = 3, epsilon = 0.2) -> torch.Tensor:
#     resample_x = np.array(x) #resample(x, N, axis=2)
#     train_data = []

#     for b in range(resample_x.shape[0]):
#         M = int((resample_x.shape[-1]-T)/S)
#         data_list = []
        
#         for i in range(M):
#             DDE_list = []
#             r_slice = resample_x[b, :, i*S:i*S+T]
#             for j in range(r_slice.shape[0]):
#                 r = r_slice[j,:]
#                 emd = EMD(energy_ratio_thr=epsilon)
#                 imf = emd(r, max_imf=iter_num)
#                 D_list = []
#                 for k in range(iter_num):
#                     try:
#                         D = 0.5 * np.log(np.sum(np.abs(imf[k])**2)) + 0.5 * np.log(2*np.pi*np.e / T)
#                     except:
#                         D = 0.5 * np.log(2*np.pi*np.e / T)
#                     D_list.append(D)
#                 DDE_list.append(D_list)
#             data_list.append(DDE_list)
#         data_list = np.transpose(data_list, (1,2,0))
#         data_list = data_list.reshape(-1, M*iter_num).T

#         train_data.append(data_list)

#     train_data = np.array(train_data)
#     train_data = torch.tensor(train_data, dtype=torch.float32)
#     print(train_data.shape)

#     return train_data

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    start = time.perf_counter() #計測開始
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)

    for train in train_loader:
        X, y, subject_idx = train
        print(X.shape, train_set.num_channels, train_set.seq_len)
        break


if __name__ == "__main__":
    run()