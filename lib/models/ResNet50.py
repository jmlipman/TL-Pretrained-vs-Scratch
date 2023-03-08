import torch
from lib.models.BaseModel import BaseModel
from torch.nn.functional import interpolate
from torch.nn import Conv3d, Conv2d, InstanceNorm2d, InstanceNorm3d
from torch.nn import ReLU, AvgPool2d, AvgPool3d
from torch.nn import ConvTranspose2d, ConvTranspose3d
import numpy as np
import os
import torchvision
from lib.models.Blocks import ConvBlock


class MyBottleneck(torch.nn.Module):

    def __init__(self, resnet_bn, layna):
        super(MyBottleneck, self).__init__()

        self.b1 = ConvBlock([resnet_bn.conv1, resnet_bn.bn1, ReLU()], layna+"_conv1")
        self.b2 = ConvBlock([resnet_bn.conv2, resnet_bn.bn2, ReLU()], layna+"_conv2")
        self.b2.setParents([self.b1])
        self.b3 = ConvBlock([resnet_bn.conv3, resnet_bn.bn3], layna+"_conv3")
        self.b3.setParents([self.b2])
        if resnet_bn.downsample is not None:
            self.dn = ConvBlock([resnet_bn.downsample], layna+"_downsample")
        else:
            self.dn = None
        self.lastrelu = ReLU()

    def forward(self, x):
        identity = x

        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)

        if self.dn is not None:
            identity = self.dn(x)

        out += identity
        out = self.lastrelu(out)

        return out

class ResNet50(BaseModel):

    # Parameters of the model
    # Some notes:
    # "parents" are set to know which input filters must be deleted in each conv
    params = ["n_classes"]
    def __init__(self, n_classes, pretrained):

        # weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        # weights = weights / weights.sum()

        super(ResNet50, self).__init__()
        self.n_classes = n_classes
        # 1) LOAD WEIGHTS IF NECESSARY
        #self.model = torchvision.models.resnet50()
        self._model = torchvision.models.resnet50(pretrained=pretrained)
        del self._model.fc # We don't need the fully connected layer.

        # 2) PRUNE

        # 3) DIVIDE AND WRAP INTO CONVBLOCKS
        # This is important because convblocks will compute the deltas,
        # and they contain a reference to each convblock parents, which is
        # later used for knowing the number of input_filters to prune

        convblock1 = ConvBlock([
                self._model.conv1,
                self._model.bn1,
                self._model.relu
                ], "conv1")
        convblock1.setParents([])
        mylayer1_bn1 = MyBottleneck(self._model.layer1[0], "layer1_0")
        mylayer1_bn1.b1.setParents([convblock1])
        mylayer1_bn1.dn.setParents([convblock1])
        mylayer1_bn2 = MyBottleneck(self._model.layer1[1], "layer1_1")
        mylayer1_bn2.b1.setParents([mylayer1_bn1.b3, mylayer1_bn1.dn])
        mylayer1_bn3 = MyBottleneck(self._model.layer1[2], "layer1_2")
        mylayer1_bn3.b1.setParents([mylayer1_bn2.b3])

        ###
        mylayer2_bn1 = MyBottleneck(self._model.layer2[0], "layer2_0")
        mylayer2_bn1.b1.setParents([mylayer1_bn3.b3])
        mylayer2_bn1.dn.setParents([mylayer1_bn3.b3])
        mylayer2_bn2 = MyBottleneck(self._model.layer2[1], "layer2_1")
        mylayer2_bn2.b1.setParents([mylayer2_bn1.b3, mylayer2_bn1.dn])
        mylayer2_bn3 = MyBottleneck(self._model.layer2[2], "layer2_2")
        mylayer2_bn3.b1.setParents([mylayer2_bn2.b3])
        mylayer2_bn4 = MyBottleneck(self._model.layer2[3], "layer2_3")
        mylayer2_bn4.b1.setParents([mylayer2_bn3.b3])

        mylayer3_bn1 = MyBottleneck(self._model.layer3[0], "layer3_0")
        mylayer3_bn1.b1.setParents([mylayer2_bn4.b3])
        mylayer3_bn1.dn.setParents([mylayer2_bn4.b3])
        mylayer3_bn2 = MyBottleneck(self._model.layer3[1], "layer3_1")
        mylayer3_bn2.b1.setParents([mylayer3_bn1.b3, mylayer3_bn1.dn])
        mylayer3_bn3 = MyBottleneck(self._model.layer3[2], "layer3_2")
        mylayer3_bn3.b1.setParents([mylayer3_bn2.b3])
        mylayer3_bn4 = MyBottleneck(self._model.layer3[3], "layer3_3")
        mylayer3_bn4.b1.setParents([mylayer3_bn3.b3])
        mylayer3_bn5 = MyBottleneck(self._model.layer3[4], "layer3_4")
        mylayer3_bn5.b1.setParents([mylayer3_bn4.b3])
        mylayer3_bn6 = MyBottleneck(self._model.layer3[5], "layer3_5")
        mylayer3_bn6.b1.setParents([mylayer3_bn5.b3])

        mylayer4_bn1 = MyBottleneck(self._model.layer4[0], "layer4_0")
        mylayer4_bn1.b1.setParents([mylayer3_bn6.b3])
        mylayer4_bn1.dn.setParents([mylayer3_bn6.b3])
        mylayer4_bn2 = MyBottleneck(self._model.layer4[1], "layer4_1")
        mylayer4_bn2.b1.setParents([mylayer4_bn1.b3, mylayer4_bn1.dn])
        mylayer4_bn3 = MyBottleneck(self._model.layer4[2], "layer4_2")
        mylayer4_bn3.b1.setParents([mylayer4_bn2.b3])

        self.encoder = torch.nn.Sequential(
                convblock1,
                self._model.maxpool,
                mylayer1_bn1,
                mylayer1_bn2,
                mylayer1_bn3,
                mylayer2_bn1,
                mylayer2_bn2,
                mylayer2_bn3,
                mylayer2_bn4,
                mylayer3_bn1,
                mylayer3_bn2,
                mylayer3_bn3,
                mylayer3_bn4,
                mylayer3_bn5,
                mylayer3_bn6,
                mylayer4_bn1,
                mylayer4_bn2,
                mylayer4_bn3,
                self._model.avgpool
                )

        self.last = torch.nn.Sequential(ConvBlock([
            torch.nn.Linear(in_features=2048, out_features=n_classes)
            ], "last"))
        self.last[0].setParents([mylayer4_bn3.b3])
        #self.last = torch.nn.Linear(in_features=2048, out_features=n_classes)

        # Because of the residual connections, when pruning certain layers, we
        # must prune the same number of filters in other layers. E.g., if we
        # prune the output channels in layer1_bn1.conv3, we must also prune
        # the output channels in layer1_bn1.downsample, layer1_bn2.conv3, and
        # layer1_bn3.conv3.
        # Here, we keep track of which layers must be have the same number of
        # filters pruned. In addition, certain layers will prune the filters
        # in the same position. For example, it makes sense that we prune
        # the same (location-wise) filters in layer1_bn1.conv3 and
        # ayer1_bn1.downsample because they will be summed. However, for layers
        # layer1_bn2.conv3 and layer1_bn3.conv3, we don't need to prune the
        # same (location-wise) filters; just prune the same *number* of filters

        # Each line indicates that the number of output filters must be the same
        # and, if they're in parenthesis, it means that their location must be
        # the same too.
        #self.encoder[0].in_channels = 3
        self.relations = [
            [(self.encoder[2].b3, self.encoder[2].dn), self.encoder[3].b3, self.encoder[4].b3],
            [(self.encoder[5].b3, self.encoder[5].dn), self.encoder[6].b3, self.encoder[7].b3, self.encoder[8].b3],
            [(self.encoder[9].b3, self.encoder[9].dn), self.encoder[10].b3, self.encoder[11].b3, self.encoder[12].b3, self.encoder[13].b3, self.encoder[14].b3],
            [(self.encoder[15].b3, self.encoder[15].dn), self.encoder[16].b3, self.encoder[17].b3],
            ]
        # TODO: When saving the model, do not save everything twice...

        # Here, I need to prune if necessary...

    def forward(self, x):

        x = x[0]
        x = self.encoder(x)
        #print(x.sum(), x.shape)
        #asdf

        outputs = torch.flatten(x, 1)
        outputs = self.last(outputs)
        if outputs.shape[1] == 2: # ISIC2020 dataset
            outputs = torch.functional.F.softmax(outputs, dim=1)
        else: # CheXpert
            outputs = torch.sigmoid(outputs)

        #return outputs # Put this when wrapping this with Sauron
        return outputs

class ResNet50_ext(BaseModel):

    params = ["n_classes"]
    def __init__(self, n_classes, pretrained):

        super(ResNet50_ext, self).__init__()
        self.n_classes = n_classes
        self._model = torchvision.models.resnet50(pretrained=pretrained)
        del self._model.fc # We don't need the fully connected layer.
        convblock1 = ConvBlock([
                self._model.conv1,
                self._model.bn1,
                self._model.relu
                ], "conv1")
        convblock1.setParents([])
        mylayer1_bn1 = MyBottleneck(self._model.layer1[0], "layer1_0")
        mylayer1_bn1.b1.setParents([convblock1])
        mylayer1_bn1.dn.setParents([convblock1])
        mylayer1_bn2 = MyBottleneck(self._model.layer1[1], "layer1_1")
        mylayer1_bn2.b1.setParents([mylayer1_bn1.b3, mylayer1_bn1.dn])
        mylayer1_bn3 = MyBottleneck(self._model.layer1[2], "layer1_2")
        mylayer1_bn3.b1.setParents([mylayer1_bn2.b3])

        ###
        mylayer2_bn1 = MyBottleneck(self._model.layer2[0], "layer2_0")
        mylayer2_bn1.b1.setParents([mylayer1_bn3.b3])
        mylayer2_bn1.dn.setParents([mylayer1_bn3.b3])
        mylayer2_bn2 = MyBottleneck(self._model.layer2[1], "layer2_1")
        mylayer2_bn2.b1.setParents([mylayer2_bn1.b3, mylayer2_bn1.dn])
        mylayer2_bn3 = MyBottleneck(self._model.layer2[2], "layer2_2")
        mylayer2_bn3.b1.setParents([mylayer2_bn2.b3])
        mylayer2_bn4 = MyBottleneck(self._model.layer2[3], "layer2_3")
        mylayer2_bn4.b1.setParents([mylayer2_bn3.b3])

        mylayer3_bn1 = MyBottleneck(self._model.layer3[0], "layer3_0")
        mylayer3_bn1.b1.setParents([mylayer2_bn4.b3])
        mylayer3_bn1.dn.setParents([mylayer2_bn4.b3])
        mylayer3_bn2 = MyBottleneck(self._model.layer3[1], "layer3_1")
        mylayer3_bn2.b1.setParents([mylayer3_bn1.b3, mylayer3_bn1.dn])
        mylayer3_bn3 = MyBottleneck(self._model.layer3[2], "layer3_2")
        mylayer3_bn3.b1.setParents([mylayer3_bn2.b3])
        mylayer3_bn4 = MyBottleneck(self._model.layer3[3], "layer3_3")
        mylayer3_bn4.b1.setParents([mylayer3_bn3.b3])
        mylayer3_bn5 = MyBottleneck(self._model.layer3[4], "layer3_4")
        mylayer3_bn5.b1.setParents([mylayer3_bn4.b3])
        mylayer3_bn6 = MyBottleneck(self._model.layer3[5], "layer3_5")
        mylayer3_bn6.b1.setParents([mylayer3_bn5.b3])

        mylayer4_bn1 = MyBottleneck(self._model.layer4[0], "layer4_0")
        mylayer4_bn1.b1.setParents([mylayer3_bn6.b3])
        mylayer4_bn1.dn.setParents([mylayer3_bn6.b3])
        mylayer4_bn2 = MyBottleneck(self._model.layer4[1], "layer4_1")
        mylayer4_bn2.b1.setParents([mylayer4_bn1.b3, mylayer4_bn1.dn])
        mylayer4_bn3 = MyBottleneck(self._model.layer4[2], "layer4_2")
        mylayer4_bn3.b1.setParents([mylayer4_bn2.b3])

        self.encoder = torch.nn.Sequential(
                convblock1,
                self._model.maxpool,
                mylayer1_bn1,
                mylayer1_bn2,
                mylayer1_bn3,
                mylayer2_bn1,
                mylayer2_bn2,
                mylayer2_bn3,
                mylayer2_bn4,
                mylayer3_bn1,
                mylayer3_bn2,
                mylayer3_bn3,
                self._model.maxpool,
                mylayer3_bn4,
                mylayer3_bn5,
                mylayer3_bn6,
                mylayer4_bn1,
                mylayer4_bn2,
                mylayer4_bn3,
                )

        self.last = torch.nn.Sequential(ConvBlock([
            torch.nn.Linear(in_features=2048, out_features=n_classes)
            ], "last"))
        self.last[0].setParents([mylayer4_bn3.b3])

        # I would need to re-do this after [12]


    def forward(self, x):

        x = x[0]
        x = self.encoder(x)
        x = self._model.avgpool(x)

        outputs = torch.flatten(x, 1)
        outputs = self.last(outputs)
        if outputs.shape[1] == 2: # ISIC2020 dataset
            outputs = torch.functional.F.softmax(outputs, dim=1)
        else: # CheXpert
            outputs = torch.sigmoid(outputs)

        return outputs

