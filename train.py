###########################################
# This script trains great UNet baselines #
# Prior to this, run preprocess.py #
###########################################

import torch, os, time, re
from lib.utils import parseArguments, Log
from torch.utils.data import DataLoader
import numpy as np
import lib.callback as callback
from lib.models.Sauron import Sauron
from lib.models.nnUNet import nnUNeXt
from torch.optim.lr_scheduler import LambdaLR
torch.autograd.set_detect_anomaly(True)

def train(cfg: dict, exp_name: str):
    # Set output path = output_path + exp_name
    cfg["path"] = os.path.join(cfg["path"], exp_name)
    # Create output folder if it doesn't exist, and find 'run ID'
    if not os.path.isdir(cfg["path"]):
        os.makedirs(cfg["path"])
        run_id = 1
    else:
        run_folders = [int(x) for x in os.listdir(cfg["path"]) if x.isdigit()]
        if len(run_folders) > 0:
            run_id = np.max(run_folders)+1
        else:
            run_id = 1
    cfg["path"] = os.path.join(cfg["path"], str(run_id))
    os.makedirs(cfg["path"])

    Log(os.path.join(cfg["path"], "config.json")).saveConfig(cfg)
    log = Log(os.path.join(cfg["path"], "log.txt"))

    # Train nnUNet
    model = Sauron(**cfg["architecture"])
    dataset = cfg["data"]

    log(f"Starting {exp_name} "
        f"(run={run_id})")

    model.initialize(cfg["device"], cfg["model_state"], log)
    data = dataset(cfg["fold"], cfg["percentage"], cfg["resolution"])
    t0 = time.time()

    if cfg["iterations"] > 0:
        tr_data = data.get("train")
        val_data = data.get("validation")

        num_samp = len(tr_data.dataset.subjects_dict["train"])
        oneEpInIters = 1+(num_samp//cfg['batch_size'])
        log(f"1 epoch = {oneEpInIters} iterations")
        howManyEpochs = np.round(cfg["iterations"]/oneEpInIters, 2)
        log(f"{cfg['iterations']} iterations = {howManyEpochs} epochs")

        if len(tr_data) > 0:
            # DataLoaders. Note that shuffle=False because I randomize it myself
            # Note that num_workers=0 because I add this in the Queue
            tr_loader = DataLoader(tr_data, batch_size=cfg["batch_size"],
                    shuffle=False, pin_memory=False, num_workers=0)

            optimizer = cfg["optim"](model.parameters(), **cfg["optim_params"])
            if cfg["model_state"] != "":
                opt_path = re.sub("models/model", "models/opt", cfg["model_state"])
                if os.path.isfile(opt_path):
                    log("Loading optimizer's state dict")
                    optimizer.load_state_dict(torch.load(opt_path))
                    #from IPython import embed; embed(); asd

            # Scheduler
            if cfg["scheduler"] == "poly":
                scheduler = LambdaLR(optimizer,
                        lr_lambda=lambda it: (1 - it/cfg["iterations"])**0.9)
            else:
                scheduler = None

            # Folder for saving the trained models
            os.makedirs(f"{cfg['path']}/models")

            model.fit(tr_loader=tr_loader, val_data=val_data,
                    iteration_start=cfg["iteration_start"],
                    iterations=cfg["iterations"], val_interval=cfg["val_interval"],
                    loss=cfg["loss"],
                    val_batch_size=cfg["batch_size"], opt=optimizer,
                    scheduler=scheduler,
                    callbacks=cfg["callbacks"],
                    log=log, history=cfg["history"])

    log(f"Total running time - {np.round((time.time()-t0)/3600, 3)} hours")
    log("End")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Gather all input arguments.
    # --dataset is mandatory
    # Dataset "files" (located in lib/data) contain their optimal data aug.
    # and optimization strategy and UNet architecture to achieve SOTA.
    # If that information is specified, the given argument will be used.
    # For example, --epochs 30 will force the training script to run for 30 epochs.
    cfg = parseArguments()

    # You can force or add custom config here. Example:
    # cfg["train.epochs"] = 999
    # cfg["train.new_config"] = "test"
    cfg["callbacks"] = [
            #callback._start_train_iteration_save_images,
            callback._end_train_iteration_save_10_models
            ]
    if cfg["architecture"]["dist_fun"] != "":
        sauron_callbacks = [
            callback._end_train_iteration_prune,
            callback._end_train_iteration_save_history,
            callback._end_train_iteration_track_number_filters,
                ]
        cfg["callbacks"] = sauron_callbacks + cfg["callbacks"]
    exp_name = f"TL_ISIC2020/{cfg['resolution']}/{cfg['architecture']['network_name']}/keep{cfg['percentage']}/fold{cfg['fold']}/"

    train(cfg, exp_name)
