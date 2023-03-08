from typing import Type, List, Dict
from lib.models.BaseModel import BaseModel, unwrap_data
import torch, os, time
import numpy as np
from torchio.data.dataset import SubjectsDataset
from torchio.data.subject import Subject
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor
import nibabel as nib
from lib.models.Sauron import DropChannels
import pickle
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler

"""
Callback functions are executed at particular times during the training or
validation of the models. The name of the callback function indicates when
it is executed. For now, all callback function's names must start with:
    * _start_epoch_: Executed at the beginning of each epoch.
    * _end_epoch_: Executed at the end of each epoch.
    * _start_train_iteration_: Executed at the beginning of each iteration.
    * _end_train_iteration_: Executed at the beginning of each iteration.
    * _start_val_subject: Executed at the beginning of each val sub iteration.
    * _end_val_subject: Executed at the end of each val sub iteration.

Callbacks' arguments must have the same name of the variables that expect.
These variables are gathered using "locals()", and passed to the callbacks.
Note that the model can be accessed with the parameter `self`.
"""

def _start_training_scheduler_init(scheduler: _LRScheduler,
        iteration_start: int):
    """
    Decreases the learning rate by using the scheduler before training starts.
    This is useful when running on servers that allow you to run jobs for a
    limited time. Thus, when continue running those jobs, learning rate needs
    to be decreased before training.

    Args:
      `scheduler`: Custom name of the lr scheduling strategy.
      `iteration_start`: Number of times scheduler.step() will be executed.

    """
    if not scheduler is None and iteration_start > 1:
        for it in range(1, iteration_start):
            scheduler.step()

def _end_train_iteration_save_history(self: Type[BaseModel], val_loss_history: List,
        tr_loss_history: List, channels_history: Dict[str, List[int]],
        outputPath: str, it: int, val_interval: int) -> None:

    if it % val_interval != 0:
        return

    with open(os.path.join(outputPath, "val_loss_history.pkl"), "wb") as f:
        pickle.dump(val_loss_history, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, "tr_loss_history.pkl"), "wb") as f:
        pickle.dump(tr_loss_history, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, "channels_history.pkl"), "wb") as f:
        pickle.dump(channels_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    patiences, thrs = {}, {} # rho, tau
    for mod in self.modules():
        if isinstance(mod, DropChannels):
            patiences[mod.name] = mod.patience
            thrs[mod.name] = mod.thr

    with open(os.path.join(outputPath, "mod_patience.pkl"), "wb") as f:
        pickle.dump(patiences, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, "mod_thr.pkl"), "wb") as f:
        pickle.dump(thrs, f, protocol=pickle.HIGHEST_PROTOCOL)


def _end_train_iteration_prune(self: Type[BaseModel], log, val_loss_history: List,
        tr_loss_history: List, opt, it: int,
        channels_history: Dict[str, List[int]], val_interval: int) -> None:

    if it % val_interval != 0:
        return

    tau_max = 0.3
    kappa = 15
    rho = 5
    mu = 2/100

    def convergence(channel_history, val_loss_history, tr_loss_history):
        if len(val_loss_history) < rho+1:
            return False

        tr_loss_history = np.array(tr_loss_history)
        val_loss_history = np.array(val_loss_history)

        # The very last loss values
        last_tr_loss = tr_loss_history[-1]
        last_val_loss = val_loss_history[-1]

        # The previous N values to the last
        prev_tr_loss = tr_loss_history[-rho-1:-1]
        #prev_val_loss = val_loss_history[-conv_N-1:-1]

        # Training loss got better, so don't increase the pruning thr
        if last_tr_loss < prev_tr_loss.min():
            print("Uno")
            return False

        # Training loss got worse, so don't increase the pruning thr
        if last_tr_loss > prev_tr_loss.max():
            print("Dos")
            return False

        # Validation loss got better, so don't increase the pruning thr
        if last_val_loss < val_loss_history.min():
            print("Tres")
            return False

        # Actually, I think this is useless, but I'm not sure.
        # The idea is that if it has not decreased too much, return true
        thr = int(np.ceil(channel_history[-2]*mu))
        if (channel_history[-2] - channel_history[-1]) < thr:
            print("Cuatro")
            return True

        print("Cinco")
        return False


    print("Enter pruning")
    #from IPython import embed; embed(); asd
    ### PART 1. Prune the channels with distance smaller than thr.
    # new_params and old_params keep track of the old and new parameters
    # which is utilized below.

    ### PART 1A. Update module's thresholds if necessary
    new_params, old_params = [], []
    for mod in self.modules():
        if isinstance(mod, DropChannels) and mod.imp_fun:
            if mod.patience > 0:
                mod.patience -= 1
            elif (mod.name in channels_history
                    #and False
                    and convergence(channels_history[mod.name],
                                    val_loss_history,
                                    tr_loss_history) ): # convergencea
                # (rho) Default patience value to avoid moving the threshold
                mod.patience = rho
                # Update threshold linearly
                thrs = np.linspace(0.001, tau_max, kappa+1) # tau_max, kappa
                if mod.thr != thrs[-1]:
                    idx = mod.thr < thrs
                    mod.thr = thrs[idx][0]

                log(f"Increasing thr to {np.round(mod.thr, 3)} in {mod.name}")

            ### PART 1B. Find which filters to prune
            mod.remove_channels = mod.delta_prune < mod.thr
            # Ignore the channel used as a reference to compute the distances
            mod.remove_channels[mod.ref_idx] = False
            #mod.remove_channels = torch.zeros(mod.delta_prune.shape, dtype=torch.bool)

    ### PART 1C. Decide which filters will be pruned in those convs that are related
    # First, convs that will have the same number and location of pruned filters.
    for i, relation in enumerate(self.network.relations):
        min_num_prune_filters = 999999
        flatten_relation = []
        for mods in relation:
            if isinstance(mods, tuple):
                idx = self.name2dropchannel[mods[0].name].remove_channels
                for m in mods[1:]:
                    idx *= self.name2dropchannel[m.name].remove_channels
                for m in mods:
                    self.name2dropchannel[m.name].remove_channels = idx
                    flatten_relation.append(m)
                total_num_idx = int(idx.sum().cpu().detach().numpy())
                min_num_prune_filters = np.min([min_num_prune_filters, total_num_idx])
            else:
                total_num_idx = int(self.name2dropchannel[mods.name].remove_channels.sum().cpu().detach().numpy())
                min_num_prune_filters = np.min([min_num_prune_filters, total_num_idx])
                flatten_relation.append(mods)
        # At this point, for this particular "relation", we know how many filters
        # we should prune -> min_num_prune_filters. Now, figure out which ones.
        for mod in flatten_relation:
            # the +1 is because one of the filters will be "pi", i.e., the
            # filter chosen w.r.t. distances are calculated that we cannot remove
            # Thus, its distance will be zero, and it will be the first of the
            # torch.argsort elements
            #from IPython import embed; embed(); asd
            #dp = torch.argsort(self.name2dropchannel[mod.name].delta_prune)[1:min_num_prune_filters+1]
            dp = torch.argsort(self.name2dropchannel[mod.name].delta_prune)[self.name2dropchannel[mod.name].remove_channels][:min_num_prune_filters]
            tt = torch.zeros_like(self.name2dropchannel[mod.name].delta_prune, dtype=torch.bool)
            tt[dp] = True
            self.name2dropchannel[mod.name].remove_channels = tt

    for mod in self.modules():
        if isinstance(mod, DropChannels) and mod.imp_fun:
            log(f"Deleting {mod.remove_channels.sum()} filters in {mod.name}; thr={mod.thr}")

    ### PART 2. Remove the filters in the encoder
    for mod in self.network.modules():
        if isinstance(mod, DropChannels):
            mod.prune(self.name2dropchannel, self.bnid2name, opt)


def _end_train_iteration_track_number_filters(self: Type[BaseModel], outputPath: str,
        channels_history: Dict[str, List[int]], it: int, val_interval: int):
    """
    Record the number of input filters in every conv. layer within
    'DropChannelWrapper' objects.
    Save all channels in `channel_history` to enable convergence detection.

    Args:
      `self`: model.
      `path_handler` (lib.utils.handlers.PathHandler).
      `channel_history`: Number of input chann. per DropChannelWrapper object.
    """
    """
    if (not hasattr(self, "dropchannelModules")
            or not isinstance(self.dropchannelModules, list)):
        message = ("The network must have an attribute named"
                "'dropchannelModule' that is of type 'list'."
                "It should contain the DropChannels modules")
        raise Exception(message)
    """

    if it % val_interval != 0:
        return

    in_filters = {}
    for mod in self.network.modules():
        if not isinstance(mod, DropChannels):
            continue
        for submod in mod.modules():
            if isinstance(submod, (torch.nn.Conv2d, torch.nn.Conv3d,
                torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                in_filters[mod.name] = submod.in_channels
                if not mod.name in channels_history:
                    channels_history[mod.name] = []
                # Before I used to save in_channels
                channels_history[mod.name].append(int(submod.out_channels))
            elif isinstance(submod, (torch.nn.Linear)):
                in_filters[mod.name] = submod.in_features
                if not mod.name in channels_history:
                    channels_history[mod.name] = []
                # Before I used to save in_channels
                channels_history[mod.name].append(int(submod.out_features))

    sorted_names = sorted(channels_history.keys())

    filePath_in = os.path.join(outputPath, "in_filters")
    if not os.path.isfile(filePath_in):
        with open(filePath_in, "w") as f:
            f.write("\t".join([n for n in sorted_names]) + "\n")

    filePath_out = os.path.join(outputPath, "out_filters")
    if not os.path.isfile(filePath_out):
        with open(filePath_out, "w") as f:
            f.write("\t".join([n for n in sorted_names]) + "\n")

    with open(filePath_out, "a") as f:
        f.write("\t".join([str(channels_history[n][-1]) for n in sorted_names]) + "\n")
    with open(filePath_in, "a") as f:
        f.write("\t".join([str(in_filters[n]) for n in sorted_names]) + "\n")


def _end_train_iteration_save_10_models(self: Type[BaseModel],
        outputPath: str, it: int, opt: Optimizer, val_interval: int,
        iterations: int) -> None:

    if it % (iterations//10) != 0:
        return

    path_models = os.path.join(outputPath, "models")
    if "nnUNeXt" in str(self.__class__):
        state_dict = self.state_dict()
        torch.save(state_dict,  f"{path_models}/model-{it}")
        return
    else:
        state_dict = self.network.state_dict()

    if not "nnunet" in str(self.network.__class__).lower():
        state_dict_k = list(self.network.state_dict())
        # Delete those keys that are introduced by me, and leave the same
        # names as in the original networks
        for k in state_dict_k:
            if k.startswith("last."):
                continue # save it as it is. don't delete anything
            if k.startswith("_model."):
                newk = k.replace("_model.", "")
                state_dict[newk] = state_dict[k]
            del state_dict[k]

    torch.save(state_dict,  f"{path_models}/model-{it}")

def _end_train_iteration_save_last_model(self: Type[BaseModel],
        outputPath: str, it: int, opt: Optimizer, val_interval: int,
        iterations: int) -> None:
    """
    Saves the current Pytorch model.

    Args:
      `self`: model.
      `outputPath`: Path to the output (e.g., exp_name/21
      `it`: Current iteration.
    """
    if it % val_interval != 0:
        return

    #path_models = path_handler.join("models")
    path_models = os.path.join(outputPath, "models")
    state_dict_k = list(self.network.state_dict())
    state_dict = self.network.state_dict()
    # Delete those keys that are introduced by me, and leave the same
    # names as in the original networks
    for k in state_dict_k:
        if k.startswith("last."):
            continue # save it as it is. don't delete anything
        if k.startswith("_model."):
            newk = k.replace("_model.", "")
            state_dict[newk] = state_dict[k]
        del state_dict[k]

    torch.save(state_dict,  f"{path_models}/model-{it}")
    torch.save(opt.state_dict(), f"{path_models}/opt-{it}")
    #print(state_dict) # This shouldn't be empty
    if it > 1 and os.path.exists(f"{path_models}/model-{it-val_interval}"):
        os.remove(f"{path_models}/model-{it-val_interval}")
        os.remove(f"{path_models}/opt-{it-val_interval}")

