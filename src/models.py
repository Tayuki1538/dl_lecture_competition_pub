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
        hid_dim: int = 128,
        hid_dim2: int = 512
    ) -> None:                  # hid_dim2,3を追加
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim2)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim2, num_classes),
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

        torch.nn.init.kaiming_uniform_(self.conv0.weight)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
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
    

class VGG(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1854,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.maxpool = nn.MaxPool2d(2, 2)

        self.blocks = nn.Sequential(
            Conv2dBlock(self.in_dim, 4, kernel_size=kernel_size, p_drop=p_drop),
            Conv2dBlock(4,8, kernel_size=kernel_size, p_drop=p_drop),
            Conv2dBlock(8,16, kernel_size=kernel_size, p_drop=p_drop),
            Conv2dBlock(16,32, kernel_size=kernel_size, p_drop=p_drop),
            Conv2dBlock(32,64, kernel_size=kernel_size, p_drop=p_drop), 
            Conv2dBlock(64,128, kernel_size=kernel_size, p_drop=p_drop),
            Conv2dBlock(128,256, kernel_size=kernel_size, p_drop=p_drop),
            Conv2dBlock(256,512, kernel_size=kernel_size, p_drop=p_drop),
        )

        self.fc0 = nn.Linear(512, out_dim)

        self.flatten = nn.Flatten()

        self.Averagepool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = Rearrange("b h w -> b 1 h w")(X)

        X = self.blocks(X)

        X = self.Averagepool(X)

        X = self.flatten(X)

        X = self.fc0(X)

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return X
    

class Conv2dBlock(nn.Module):
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

        self.conv0 = nn.Conv2d(in_dim, out_dim, (kernel_size, kernel_size), padding="same")
        self.conv1 = nn.Conv2d(out_dim, out_dim, (kernel_size, kernel_size), padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")

        torch.nn.init.kaiming_uniform_(self.conv0.weight)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.batchnorm0 = nn.BatchNorm2d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_dim)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
        
        X = self.maxpool(X)

        return self.dropout(X)