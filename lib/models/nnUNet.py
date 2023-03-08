import torch
import torch.nn as nn
from lib.models.BaseModel import BaseModel
from torch.nn.functional import interpolate
from torch.nn import Conv2d, InstanceNorm2d, BatchNorm2d, AdaptiveAvgPool2d
from torch.nn import LeakyReLU, AvgPool2d
import numpy as np
import os
from lib.models.Blocks import ConvBlock
import torch.nn.functional as F

class nnUNet(BaseModel):
    # Actual differences wrt nnUNet
    # - Maxpooling instead of strided convs
    # - No maximum of feature maps: "To limit the final model size,
    #    the number of feature maps is additionally capped at 320 and 512
    #    for 3D and 2D U-Nets, respectively."

    # Parameters of the model
    params = ["modalities", "n_classes"]
    def __init__(self, modalities, n_classes, fms_init=48, levels=5,
            fms_max=480, filters={}):

        # weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        # weights = weights / weights.sum()

        super(nnUNet, self).__init__()
        self.modalities = modalities
        self.n_classes = n_classes
        self.fms_init = fms_init
        self.levels = levels

        Conv = Conv2d
        Norm = BatchNorm2d

        # Determine the number of input and output channels in each conv
        if len(filters) == 0:
            filters["in"], filters["out"] = {}, {}

            filters["in"]["enc_ConvBlock_1"] = modalities
            filters["out"]["enc_ConvBlock_1"] = fms_init
            filters["in"]["enc_ConvBlock_2"] = fms_init
            filters["out"]["enc_ConvBlock_2"] = fms_init

            for i in range(1, levels):
                filters["in"][f"enc_ConvBlock_{i*2+1}"] = filters["out"][f"enc_ConvBlock_{(i-1)*2+1}"]
                fs = np.clip(filters["in"][f"enc_ConvBlock_{i*2}"]*2, 0, fms_max)
                filters["out"][f"enc_ConvBlock_{i*2+1}"] = fs
                filters["in"][f"enc_ConvBlock_{i*2+2}"] = fs
                filters["out"][f"enc_ConvBlock_{i*2+2}"] = fs

        # Encoder
        """
        self.encoder = [ConvBlock(filters["in"]["enc_ConvBlock_1"],
                                  filters["out"]["enc_ConvBlock_1"], dim),
                        ConvBlock(filters["in"]["enc_ConvBlock_2"],
                                  filters["out"]["enc_ConvBlock_2"], dim)]
        """

        self.encoder = []
        self.encoder.append( ConvBlock(
            [Conv(filters["in"][f"enc_ConvBlock_1"],
                filters["out"][f"enc_ConvBlock_1"],
                3, padding=1, stride=1),
             Norm(filters["out"][f"enc_ConvBlock_1"]),
             LeakyReLU()
             ], name=f"enc_ConvBlock_1" ) )

        self.encoder.append( ConvBlock(
            [Conv(filters["in"][f"enc_ConvBlock_2"],
                filters["out"][f"enc_ConvBlock_2"],
                3, padding=1, stride=1),
             Norm(filters["out"][f"enc_ConvBlock_2"]),
             LeakyReLU()
             ], name=f"enc_ConvBlock_2" ) )


        #fs = [fms_init] # filters
        for i in range(1, levels):
            #fs.append ( np.clip(fs[-1]*2, 0, fms_max) )
            #self.encoder.append( ConvBlock(fs[-2], fs[-1], dim, strides=2) )
            #self.encoder.append( ConvBlock(fs[-1], fs[-1], dim) )
            self.encoder.append( ConvBlock(
                [Conv(filters["in"][f"enc_ConvBlock_{i*2+1}"],
                    filters["out"][f"enc_ConvBlock_{i*2+1}"],
                    3, padding=1, stride=2),
                 Norm(filters["out"][f"enc_ConvBlock_{i*2+1}"]),
                 LeakyReLU()
                 ], name=f"enc_ConvBlock_{i*2+1}" ) )

            self.encoder.append( ConvBlock(
                [Conv(filters["in"][f"enc_ConvBlock_{i*2+2}"],
                    filters["out"][f"enc_ConvBlock_{i*2+2}"],
                    3, padding=1, stride=1),
                 Norm(filters["out"][f"enc_ConvBlock_{i*2+2}"]),
                 LeakyReLU()
                 ], name=f"enc_ConvBlock_{i*2+2}" ) )

        self.encoder = torch.nn.Sequential(*self.encoder)

        self.avgpool = AdaptiveAvgPool2d((1, 1))

        # Linear
        self.last = torch.nn.Sequential(ConvBlock([
            torch.nn.Linear(in_features=fms_max, out_features=n_classes)
            ], "last"))

    def forward(self, x):

        x = x[0]
        output = self.encoder(x)
        #from IPython import embed; embed(); asd
        output = self.avgpool(output)
        outputs = torch.flatten(output, 1)
        outputs = self.last(outputs)
        if outputs.shape[1] == 2: # ISIC2020 dataset
            outputs = torch.functional.F.softmax(outputs, dim=1)
        else: # CheXpert
            outputs = torch.sigmoid(outputs)

        return outputs

class nnUNetv2(BaseModel):
    # Actual differences wrt nnUNet
    # - Maxpooling instead of strided convs
    # - No maximum of feature maps: "To limit the final model size,
    #    the number of feature maps is additionally capped at 320 and 512
    #    for 3D and 2D U-Nets, respectively."

    # Parameters of the model
    params = ["modalities", "n_classes"]
    def __init__(self, modalities, n_classes, fms_init=48, levels=5,
            fms_max=480, filters={}):

        # weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        # weights = weights / weights.sum()

        super(nnUNetv2, self).__init__()
        raise Exception("Not in use")
        self.modalities = modalities
        self.n_classes = n_classes
        self.fms_init = fms_init
        self.levels = levels

        Conv = Conv2d
        Norm = BatchNorm2d

        # Determine the number of input and output channels in each conv
        if len(filters) == 0:
            filters["in"], filters["out"] = {}, {}

            filters["in"]["enc_ConvBlock_1"] = modalities
            filters["out"]["enc_ConvBlock_1"] = fms_init
            filters["in"]["enc_ConvBlock_2"] = fms_init
            filters["out"]["enc_ConvBlock_2"] = fms_init
            filters["in"]["enc_ConvBlock_3"] = fms_init
            filters["out"]["enc_ConvBlock_3"] = fms_init

            for i in range(1, levels):
                filters["in"][f"enc_ConvBlock_{i*3+1}"] = filters["out"][f"enc_ConvBlock_{(i-1)*3+1}"]
                fs = np.clip(filters["in"][f"enc_ConvBlock_{i*3}"]*2, 0, fms_max)
                filters["out"][f"enc_ConvBlock_{i*3+1}"] = fs
                filters["in"][f"enc_ConvBlock_{i*3+2}"] = fs
                filters["out"][f"enc_ConvBlock_{i*3+2}"] = fs
                filters["in"][f"enc_ConvBlock_{i*3+3}"] = fs
                filters["out"][f"enc_ConvBlock_{i*3+3}"] = fs

        # Encoder
        """
        self.encoder = [ConvBlock(filters["in"]["enc_ConvBlock_1"],
                                  filters["out"]["enc_ConvBlock_1"], dim),
                        ConvBlock(filters["in"]["enc_ConvBlock_2"],
                                  filters["out"]["enc_ConvBlock_2"], dim)]
        """

        self.encoder = []
        self.encoder.append( ConvBlock(
            [Conv(filters["in"][f"enc_ConvBlock_1"],
                filters["out"][f"enc_ConvBlock_1"],
                3, padding=1, stride=1),
             Norm(filters["out"][f"enc_ConvBlock_1"]),
             LeakyReLU()
             ], name=f"enc_ConvBlock_1" ) )

        self.encoder.append( ConvBlock(
            [Conv(filters["in"][f"enc_ConvBlock_2"],
                filters["out"][f"enc_ConvBlock_2"],
                3, padding=1, stride=1),
             Norm(filters["out"][f"enc_ConvBlock_2"]),
             LeakyReLU()
             ], name=f"enc_ConvBlock_2" ) )

        self.encoder.append( ConvBlock(
            [Conv(filters["in"][f"enc_ConvBlock_3"],
                filters["out"][f"enc_ConvBlock_3"],
                3, padding=1, stride=1),
             Norm(filters["out"][f"enc_ConvBlock_3"]),
             LeakyReLU()
             ], name=f"enc_ConvBlock_3" ) )


        #fs = [fms_init] # filters
        for i in range(1, levels):
            #fs.append ( np.clip(fs[-1]*2, 0, fms_max) )
            #self.encoder.append( ConvBlock(fs[-2], fs[-1], dim, strides=2) )
            #self.encoder.append( ConvBlock(fs[-1], fs[-1], dim) )
            self.encoder.append( ConvBlock(
                [Conv(filters["in"][f"enc_ConvBlock_{i*3+1}"],
                    filters["out"][f"enc_ConvBlock_{i*3+1}"],
                    3, padding=1, stride=2),
                 Norm(filters["out"][f"enc_ConvBlock_{i*3+1}"]),
                 LeakyReLU()
                 ], name=f"enc_ConvBlock_{i*3+1}" ) )

            self.encoder.append( ConvBlock(
                [Conv(filters["in"][f"enc_ConvBlock_{i*3+2}"],
                    filters["out"][f"enc_ConvBlock_{i*3+2}"],
                    3, padding=1, stride=1),
                 Norm(filters["out"][f"enc_ConvBlock_{i*3+2}"]),
                 LeakyReLU()
                 ], name=f"enc_ConvBlock_{i*3+2}" ) )

            self.encoder.append( ConvBlock(
                [Conv(filters["in"][f"enc_ConvBlock_{i*3+3}"],
                    filters["out"][f"enc_ConvBlock_{i*3+3}"],
                    3, padding=1, stride=1),
                 Norm(filters["out"][f"enc_ConvBlock_{i*3+3}"]),
                 LeakyReLU()
                 ], name=f"enc_ConvBlock_{i*3+3}" ) )

        self.encoder = torch.nn.Sequential(*self.encoder)

        self.avgpool = AdaptiveAvgPool2d((1, 1))

        # Linear
        self.last = torch.nn.Sequential(ConvBlock([
            torch.nn.Linear(in_features=fms_max, out_features=n_classes)
            ], "last"))

    def forward(self, x):

        x = x[0]
        output = self.encoder(x)
        #from IPython import embed; embed(); asd
        output = self.avgpool(output)
        outputs = torch.flatten(output, 1)
        outputs = self.last(outputs)
        outputs = torch.sigmoid(outputs)

        return outputs

class ConvNextBlock_orig(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        #self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x1 = self.dwconv(x)
        x2 = x1.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x3 = self.norm(x2)
        x4 = self.pwconv1(x3)
        x5 = self.act(x4)
        x6 = self.pwconv2(x5)
        x7 = x6.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        #from IPython import embed; embed(); asd
        x8 = input + x7
        return x8

class ConvNextBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        #self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim*4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim*4, dim, kernel_size=1)

    def forward(self, x):
        input = x
        x1 = self.dwconv(x)
        x3 = self.norm(x1)
        x4 = self.pwconv1(x3)
        x5 = self.act(x4)
        x6 = self.pwconv2(x5)
        x7 = input + x6
        return x7

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class nnUNeXt(BaseModel):
    # Actual differences wrt nnUNet
    # - Maxpooling instead of strided convs
    # - No maximum of feature maps: "To limit the final model size,
    #    the number of feature maps is additionally capped at 320 and 512
    #    for 3D and 2D U-Nets, respectively."

    # Parameters of the model
    params = ["n_classes"]
    def __init__(self, n_classes, levels, in_fi):

        # weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        # weights = weights / weights.sum()

        super(nnUNeXt, self).__init__()

        self.encoder = []
        self.encoder.append(nn.Sequential(
            nn.Conv2d(3, in_fi, kernel_size=2, stride=2),
            #LayerNorm(in_fi, data_format="channels_first")
            nn.BatchNorm2d(in_fi)
            ))

        for i in range(levels):
            self.encoder.append(ConvNextBlock(in_fi*(2**i)))
            self.encoder.append(ConvNextBlock(in_fi*(2**i)))
            self.encoder.append(nn.Sequential( # Downsample
                #LayerNorm(in_fi*(2**i), data_format="channels_first"),
                nn.BatchNorm2d(in_fi*(2**i)),
                nn.Conv2d(in_fi*(2**i), in_fi*(2**(i+1)), kernel_size=2, stride=2),
                ))

        self.encoder = torch.nn.Sequential(*self.encoder)

        self.avgpool = AdaptiveAvgPool2d((1, 1))

        # Linear
        self.last = torch.nn.Linear(in_features=2048, out_features=n_classes)

    def forward(self, x):

        x = x[0]
        output = self.encoder(x)
        #from IPython import embed; embed(); asd
        output = self.avgpool(output)
        outputs = torch.flatten(output, 1)
        outputs = self.last(outputs)
        outputs = torch.sigmoid(outputs)

        return (outputs, )
