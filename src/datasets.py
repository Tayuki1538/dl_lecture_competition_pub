import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
from PyEMD import EMD
import time
from concurrent.futures import ThreadPoolExecutor
from einops.layers.torch import Rearrange

def np_log(x):
    return np.log(x + 1e-10)


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    # 重すぎて却下．
    def preprocessing(self, x: torch.Tensor, S = 20, T = 100, iter_num = 3, epsilon = 0.2) -> torch.Tensor: #追加
        # resample_x = np.array(x) #resample(x, N, axis=2)

        M = int((x.shape[-1]-T)/S)
        data_list = []
        
        for i in range(M):
            DDE_list = []
            r_slice = x[:, i*S:i*S+T]
            for j in range(r_slice.shape[0]):
                r = r_slice[j,:]
                emd = EMD(energy_ratio_thr=epsilon)
                imf = emd(r, max_imf=iter_num)
                D_list = []
                for k in range(iter_num):
                    try:
                        D = 0.5 * np_log(np.sum(np.abs(imf[k])**2)) + 0.5 * np_log(2*np.pi*np.e / T)
                    except:
                        D = 0.5 * np_log(2*np.pi*np.e / T)
                    D_list.append(D)
                DDE_list.append(D_list)
            data_list.append(DDE_list)
        data_list = np.transpose(data_list, (1,2,0))
        data_list = data_list.reshape(-1, M*iter_num).T

        train_data = torch.tensor(data_list, dtype=torch.float32)

        return train_data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        # cprint(f"Loading {self.split} data: {i} / {self.num_samples}", "green") # 時間を計りたい時に使用する．
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))
        # X = self.preprocessing(np.load(X_path))

        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]