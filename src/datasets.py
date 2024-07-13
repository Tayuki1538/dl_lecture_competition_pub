import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample
from PyEMD import EMD

def np_log(x):
    return np.log(x + 1e-10)


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.X = self.preprocessing(self.X) # 追加
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def preprocessing(self, x: torch.Tensor, S = 20, T = 100, iter_num = 3, epsilon = 0.2) -> torch.Tensor: #追加
        resample_x = np.array(x) #resample(x, N, axis=2)
        train_data = []

        for b in range(resample_x.shape[0]):
            M = int((resample_x.shape[-1]-T)/S)
            data_list = []
            
            for i in range(M):
                DDE_list = []
                r_slice = resample_x[b, :, i*S:i*S+T]
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

            train_data.append(data_list)

        train_data = np.array(train_data)
        train_data = torch.tensor(train_data, dtype=torch.float32)

        return train_data
    

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]