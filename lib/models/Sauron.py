import torch
from lib.models.BaseModel import BaseModel
from torch.nn.functional import interpolate
from torch.nn import Conv3d, Conv2d, InstanceNorm2d, InstanceNorm3d
from torch.nn import LeakyReLU, AvgPool2d, AvgPool3d
from torch.nn import ConvTranspose2d, ConvTranspose3d
import numpy as np
import os
from lib.models.ResNet50 import ResNet50, MyBottleneck, ResNet50_ext
from lib.models.nnUNet import nnUNet, nnUNetv2
from lib.models.MobileNetv2 import MobileNetv2
import lib.distance as distance
import pandas as pd
from lib.models.Blocks import ConvBlock

distances = {
        "euc_norm": distance.Euclidean_norm,
        "euc_norm_nonorm": distance.Euclidean_norm_nonorm,
        "euc_norm_deltaprunenorm": distance.Euclidean_norm_deltaprunenorm,
        "euc_rand": distance.Euclidean_rand,
        "": None,
        }

def decimate(net, per):
    filters = {
            "in": {"conv1": 3, "last": int(2048*per)},
            "out": {"conv1": int(64*per), "last": 14}}
    for mm in net.modules():
        if not isinstance(mm, MyBottleneck):
            continue
        for m in mm.modules():
            if not isinstance(m, ConvBlock):
                continue
            if isinstance(m.blockmodules[0], torch.nn.Sequential):
                filters["in"][m.name] = int(m.blockmodules[0][0].in_channels*per)
                filters["out"][m.name] = int(m.blockmodules[0][0].out_channels*per)
            else:
                filters["in"][m.name] = int(m.blockmodules[0].in_channels*per)
                filters["out"][m.name] = int(m.blockmodules[0].out_channels*per)
    return filters


def wrap_convblocks_into_dropchannels(model, settings):
    name2dropchannel = {}
    for n, module in model.named_children():

        if len(list(module.children())) > 0:
            name2dropchannel.update(
                    wrap_convblocks_into_dropchannels(module, settings))

        if isinstance(module, ConvBlock):
            dist_fun, imp_fun, compress = settings
            params = { "module": module,
                       "name": module.name, "parents": module.blockparents,
                       "dist_fun": dist_fun, "imp_fun": imp_fun,
                       "compress": compress}
            setattr(model, n, DropChannels(**params))
            name2dropchannel[module.name] = getattr(model, n)
    return name2dropchannel

class Sauron(BaseModel):
    """
    Sauron wraps an existing neural network.
    Inside Sauron's initialization, we wrap blocks with DropChannel.
    """

    # Parameters of the model
    params = ["network_name", "n_classes"]
    def __init__(self, network_name, n_classes, pretrained: bool,
            dist_fun: str="", imp_fun: str="", sf: int=2,
            filters={}):
        super(Sauron, self).__init__()
        # We save these properties (mandatory for logging)
        self.n_classes = n_classes

        """
        path_history = ""
        if path_history == "":
            filters = {}
        else:
            df_in = pd.read_csv(os.path.join(path_history, "in_filters"), sep="\t")
            df_out = pd.read_csv(os.path.join(path_history, "out_filters"), sep="\t")
            filters = {}
            filters["in"] = {col_name: df_in[col_name].iloc[-1] for col_name in df_in.columns}
            filters["out"] = {col_name: df_out[col_name].iloc[-1] for col_name in df_out.columns}
        """
        # Initialize the CNN
        if network_name == "resnet50":
            self.network = ResNet50(n_classes, pretrained)
        elif network_name == "resnet50_25":
            self.network = ResNet50(n_classes, pretrained)
            filters = decimate(self.network, 0.25)
        elif network_name == "resnet50_50":
            self.network = ResNet50(n_classes, pretrained)
            filters = decimate(self.network, 0.5)
        elif network_name == "resnet50_ext1":
            self.network = ResNet50_ext(n_classes, pretrained)
        elif network_name == "nnunet_encoder_v2": # 23M params
            self.network = nnUNet(3, n_classes,
                    fms_init=64, levels=7, fms_max=600)
        else:
            raise Exception(f"Network name `{network_name}` is unknown.")

        # Wrap each "block" with DropChannels. The following code will go
        # through different blocks within nnUNet and will wrap them.
        # Important information to wrap each code:
        # - name: name of a block, for debugging reasons.
        # - parents: list of blocks that will provide the input
        # - dist_fun: distance that will be minimized (delta_opt)
        # - imp_fun: distance that will be used for thresholding (delta_prune)

        dist_fun = distances[dist_fun]
        imp_fun = distances[imp_fun]
        compress = torch.nn.AvgPool2d(sf)

        self.name2dropchannel = wrap_convblocks_into_dropchannels(self.network.encoder, (dist_fun, imp_fun, compress))
        # For the "last" module, I don't want to compute distances because
        # I dont want to prune output channels, which correspond to the classes
        # although I still want to prune the number of input channels.
        self.name2dropchannel.update(wrap_convblocks_into_dropchannels(self.network.last, (None, None, compress)))

        # Note: this assumes that there is a single batchnorm within a convblock
        self.bnid2name = {}
        for mod in self.modules():
            #if isinstance(mod, DropChannels) and mod.imp_fun:
            if isinstance(mod, DropChannels):
                for m in mod.modules():
                    if isinstance(m, (torch.nn.BatchNorm2d,
                                        torch.nn.BatchNorm3d)):
                        self.bnid2name[id(m)] = mod.name

        if len(filters) > 0: # Loading a pruned model
            for mod_name, mod in self.name2dropchannel.items():
                fi_in = filters["in"][mod_name]
                fi_out = filters["out"][mod_name]
                mod.prune(self.name2dropchannel, self.bnid2name,
                        in_filters=fi_in, out_filters=fi_out)


    def forward(self, x):

        outputs = self.network(x)

        # Gather here the distances
        distances = []
        for mod in self.network.encoder.modules():
            if isinstance(mod, DropChannels):
                distances.append(mod.delta_opt)

        return (outputs, distances)

class DropChannels(torch.nn.Module):
    """
    DropChannels must wrap a module that contains a convolution or a sequence,
    typically containing convolution+norm+act.
    """
    thr = 0.001 # Initial thresholds
    patience = 0 # To avoid pruning too fast
    def __init__(self, module, name: str, parents: list,
            dist_fun, imp_fun, compress):
        super(DropChannels, self).__init__()
        self.name = name
        self.module = module
        self.parents = parents
        self.dist_fun = dist_fun
        self.imp_fun = imp_fun
        self.compress = compress

    def forward(self, x):
        out = self.module(x)

        self.delta_opt = torch.ones((out.shape[0], 1)).cuda()
        self.delta_prune = torch.ones((1)).cuda()
        self.ref_idx = 0

        # If it's 1, there is nothing to prune
        if self.dist_fun and out.shape[1] > 1:
            self.delta_opt = self.dist_fun(out, self.compress)

        if self.imp_fun and out.shape[1] > 1:
            self.delta_prune, self.ref_idx = self.imp_fun(out, self.compress)

        return out

    def prune(self, n2d, bnid2name, opt=None, in_filters=None, out_filters=None):
        """
        This is the core of Sauron.
        If opt=None, it means that we are loading a pruned models, so,
        in/out_filters must be ints.
        """
        # n2d: dictionary that maps names (str) to dropchannels
        #

        #if len(self.parents) == 0:
        #    parent_remove_channels = torch.zeros(self.module.in_channels, dtype=torch.bool)
        #elif opt is None:
        #    parent_remove_channels = torch.zeros(self.module.in_channels, dtype=torch.bool)
        #    parent_remove_channels[:in_filters] = True
        #else:
        #    parent_remove_channels = n2d[self.parents[0].name].remove_channels
        new_params, old_params = [], []
        with torch.no_grad():
            for inner_mod in self.module.modules():
                # If shape[0] == 1, the script has skipped the code
                # that produced new mod.avg_distance_across_batches
                # and, as a consequence, the mod.avg... will be the
                # same from the prev. iteration, and since
                # it was pruned (that's why shape[0] == 1 now),
                # it will raise an error as the shape is diff.
                if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
                    if len(self.parents) == 0:
                        parent_remove_channels = torch.zeros(inner_mod.in_channels, dtype=torch.bool)
                    else:
                        parent_remove_channels = n2d[self.parents[0].name].remove_channels

                    if opt is None:
                        parent_remove_channels = torch.ones(inner_mod.in_channels, dtype=torch.bool)
                        parent_remove_channels[:in_filters] = False
                        self.remove_channels = torch.ones(inner_mod.out_channels, dtype=torch.bool)
                        self.remove_channels[:out_filters] = False
                    #torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):

                    tmp_param_weight = inner_mod.weight
                    old_params.append(id(tmp_param_weight))
                    inner_mod.weight = torch.nn.Parameter(
                            tmp_param_weight[~self.remove_channels][:, ~parent_remove_channels])
                    inner_mod.out_channels = torch.sum(
                            ~self.remove_channels).cpu().detach().numpy()
                    inner_mod.in_channels = int(torch.sum(
                            ~parent_remove_channels).cpu().detach().numpy())
                    new_params.append(inner_mod.weight)

                    if inner_mod.bias: # sometimes bias=False
                        tmp_param_bias = inner_mod.bias
                        old_params.append(id(tmp_param_bias))
                        inner_mod.bias = torch.nn.Parameter(
                                tmp_param_bias[~self.remove_channels])
                        new_params.append(inner_mod.bias)

                    # Keep step, exp_avg, and exp_avg_sq (for Adam
                    # optimizer). This depends on the optimizer.
                    # If SGD + momentum is utilized then you can
                    # replace the six lines below with these two:
                    # self.opt.state[inner_mod.weight]["momentum_buffer"] = self.opt.state[tmp_param_weight]["momentum_buffer"]
                    #self.opt.state[inner_mod.bias]["momentum_buffer"] = self.opt.state[tmp_param_bias]["momentum_buffer"]
                    if opt:
                        opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                        opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][~self.remove_channels][:, ~parent_remove_channels]
                        opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][~self.remove_channels][:, ~parent_remove_channels]
                        del opt.state[tmp_param_weight]
                    del tmp_param_weight

                    if inner_mod.bias:
                        if opt:
                            opt.state[inner_mod.bias]["step"] = opt.state[tmp_param_bias]["step"]
                            opt.state[inner_mod.bias]["exp_avg"] = opt.state[tmp_param_bias]["exp_avg"][~self.remove_channels]
                            opt.state[inner_mod.bias]["exp_avg_sq"] = opt.state[tmp_param_bias]["exp_avg_sq"][~self.remove_channels]
                            del opt.state[tmp_param_bias]
                        del tmp_param_bias

                elif isinstance(inner_mod, (torch.nn.BatchNorm2d,
                                            torch.nn.BatchNorm3d)):

                    dp = n2d[bnid2name[id(inner_mod)]]
                    tmp_param_weight = inner_mod.weight
                    old_params.append(id(tmp_param_weight))
                    inner_mod.weight = torch.nn.Parameter(
                            tmp_param_weight[~dp.remove_channels])

                    tmp_param_bias = inner_mod.bias
                    old_params.append(id(tmp_param_bias))
                    inner_mod.bias = torch.nn.Parameter(
                            tmp_param_bias[~dp.remove_channels])

                    inner_mod.num_features = torch.sum(
                            ~dp.remove_channels).cpu().detach().numpy()

                    new_params.append(inner_mod.weight)
                    new_params.append(inner_mod.bias)

                    inner_mod.running_mean = inner_mod.running_mean[~dp.remove_channels]
                    inner_mod.running_var = inner_mod.running_var[~dp.remove_channels]

                    #if self.name == "conv1":
                    #    from IPython import embed; embed(); asd

                    if opt:
                        opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                        opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][~dp.remove_channels]
                        opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][~dp.remove_channels]
                        opt.state[inner_mod.bias]["step"] = opt.state[tmp_param_bias]["step"]
                        opt.state[inner_mod.bias]["exp_avg"] = opt.state[tmp_param_bias]["exp_avg"][~dp.remove_channels]
                        opt.state[inner_mod.bias]["exp_avg_sq"] = opt.state[tmp_param_bias]["exp_avg_sq"][~dp.remove_channels]
                        del opt.state[tmp_param_weight]
                        del opt.state[tmp_param_bias]

                    #from IPython import embed; embed(); asd
                    del tmp_param_weight
                    del tmp_param_bias

                elif isinstance(inner_mod, torch.nn.Linear):
                    if len(self.parents) == 0:
                        parent_remove_channels = torch.zeros(inner_mod.in_features, dtype=torch.bool)
                    else:
                        parent_remove_channels = n2d[self.parents[0].name].remove_channels

                    if opt is None:
                        parent_remove_channels = torch.ones(inner_mod.in_features, dtype=torch.bool)
                        parent_remove_channels[:in_filters] = False

                    if self.imp_fun:
                        # NOTE: In this case, it won't load the filters properly
                        # to do that, add a self.remove_channels as in a few lines before, inside the elif
                        raise Exception("This is needed when a non-last layer that needs to be pruned is Linear")

                    else:
                        tmp_param_weight = inner_mod.weight
                        tmp_param_bias = inner_mod.bias
                        old_params.append(id(tmp_param_weight))
                        old_params.append(id(tmp_param_bias))
                        inner_mod.weight = torch.nn.Parameter(
                                tmp_param_weight[:, ~parent_remove_channels])
                        inner_mod.in_features = torch.sum(
                                ~parent_remove_channels).cpu().detach().numpy()
                        new_params.append(inner_mod.weight)
                        new_params.append(inner_mod.bias)
                        if opt:
                            opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                            opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][:, ~parent_remove_channels]
                            opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][:, ~parent_remove_channels]
                            del opt.state[tmp_param_weight]
                        del tmp_param_weight
                        del tmp_param_bias

                elif isinstance(inner_mod, (torch.nn.ConvTranspose2d,
                                            torch.nn.ConvTranspose3d)):
                    raise Exception("Is this even in use???")
                    tmp_param_weight = inner_mod.weight
                    old_params.append(id(tmp_param_weight))
                    inner_mod.weight = torch.nn.Parameter(
                            tmp_param_weight[parent_remove_channels][:, ~mod.remove_channels])
                    inner_mod.out_channels = torch.sum(
                            ~mod.remove_channels).cpu().detach().numpy()
                    inner_mod.in_channels = int(torch.sum(
                            ~parent_remove_channels).cpu().detach().numpy())
                    new_params.append(inner_mod.weight)

                    if inner_mod.bias:
                        tmp_param_bias = inner_mod.bias
                        old_params.append(id(tmp_param_bias))
                        inner_mod.bias = torch.nn.Parameter(
                                tmp_param_bias[~mod.remove_channels])
                        new_params.append(inner_mod.bias)

                    opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                    opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][~parent_remove_channels][:, ~mod.remove_channels]
                    opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][~parent_remove_channels][:, ~mod.remove_channels]
                    del opt.state[tmp_param_weight]
                    del tmp_param_weight

                    if inner_mod.bias:
                        opt.state[inner_mod.bias]["step"] = opt.state[tmp_param_bias]["step"]
                        opt.state[inner_mod.bias]["exp_avg"] = opt.state[tmp_param_bias]["exp_avg"][~mod.remove_channels]
                        opt.state[inner_mod.bias]["exp_avg_sq"] = opt.state[tmp_param_bias]["exp_avg_sq"][~mod.remove_channels]
                        del opt.state[tmp_param_bias]
                        del tmp_param_bias
        ### PART 2B. Remove the filters in the last layer, which is presumably Linear

        #from IPython import embed; embed(); assd
        # After the weights and their corresponding optimizer statistics are pruned
        # the optimizer needs to know which parameters are going to be optimized.
        # Such parameters are stored in ["params"].
        # You could tempted of doing this:
        #   self.opt.param_groups[0]["params"] = new_params
        #
        # However, new_params only contains the parameters from DropChannelWrapper
        # with mod.prune_mod = True. In other words, it does *not* contain other
        # parameters that you might have not wanted to prune (e.g., imagine
        # there is a specific layer that you didn't want to prune for some reason).
        # For this reason, this script saved old_params to not lose them.
        if not opt is None:
            replace_params = []
            for p in opt.param_groups[0]["params"]:
                # Keeps old parameters that were not subject to pruning.
                if not id(p) in old_params:
                    replace_params.append(p)
            # Add the rest of the parameters that were subject to pruning.
            replace_params += new_params
            opt.param_groups[0]["params"] = replace_params

        #if self.name == "conv1":
        #    self.module.blockmodules[1].bias = torch.nn.Parameter(torch.zeros(19))
        #    self.module.blockmodules[0].ias = torch.nn.Parameter(torch.zeros(19))
