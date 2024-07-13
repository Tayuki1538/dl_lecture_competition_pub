import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)

    
# Subject-Independent Emotion Recognition of EEG Signals Based on Dynamic Empirical Convolutional Neural Network        #追加
class DECNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1854,
        kernel_size: int = 5,
        kernel_size2: int = 3,
        kernel_size3: int = 1,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv2d(in_dim, 32, (kernel_size, kernel_size))
        self.conv1 = nn.Conv2d(32, 64, (kernel_size, kernel_size))
        self.conv2 = nn.Conv2d(64, 128, (kernel_size2, kernel_size))
        self.conv3 = nn.Conv2d(128, 256, (kernel_size2, kernel_size))
        self.conv4 = nn.Conv2d(256, 512, (kernel_size3, kernel_size))
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")

        self.maxpool = nn.MaxPool2d(2, 2)
        self.Averagepool = nn.AdaptiveAvgPool1d(1)
        
        self.batchnorm0 = nn.BatchNorm2d(num_features=32)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)

        self.dropout = nn.Dropout(p_drop)

        self.fc0 = nn.Linear(64, 1024)
        self.fc1 = nn.Linear(1024, out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.maxpool(X)

        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))
        X = self.maxpool(X)

        # X = self.conv2(X)
        # X = F.relu(X)
        # X = self.maxpool(X)

        # X = self.conv3(X)
        # X = F.relu(X)
        # X = self.maxpool(X)

        # X = self.conv4(X)
        # X = F.relu(X)
        # X = self.maxpool(X)

        X = self.Averagepool(X)

        X = X.flatten()

        X = self.fc0(X)
        X = self.fc1(X)

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return F.softmax(X)