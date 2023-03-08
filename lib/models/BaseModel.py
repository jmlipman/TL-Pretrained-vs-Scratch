from tensorboardX import SummaryWriter
import torch, os, time, json, inspect, re, pickle
from datetime import datetime
import numpy as np
import torchio as tio
#from IPython import embed
from lib.metric import Metric
from lib.utils import he_normal
from typing import List, Callable, Type, Tuple
from lib.data.BaseDataset import BaseDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import Tensor
from torchio.data.dataset import SubjectsDataset
from torchio.data.subject import Subject
import nibabel as nib
from sklearn.metrics import roc_auc_score
from torch.cuda import amp
import pandas as pd


def callCallbacks(callbacks: List[Callable], prefix: str,
        allvars: dict) -> None:
    """
    Call all callback functions starting with a given prefix.
    Check which inputs the callbacks need, and, from `allvars` (that contains
    locals()) pass those inputs.
    Read more about callback functions in lib.utils.callbacks

    Args:
      `callbacks`: List of callback functions.
      `prefix`: Prefix of the functions to be called.
      `allvars`: locals() containing all variables in memory.
    """
    for c in callbacks:
        if c.__name__.startswith(prefix):
            input_params = inspect.getfullargspec(c).args
            required_params = {k: allvars[k] for k in input_params}
            c(**required_params)


def unwrap_data(subjects_batch: dict, data: Type[BaseDataset],
        device: str) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    Extract X (input data), Y (labels), and W (weights, optional) from
    subects_batch, which comes from the tr/val_loader.

    Args:
      `subjects_batch`: Contains the train/val/test data.
      `data`: Dataset. For extracting the dimension and infoXYW.
      `device`: Device where the computations will be performed ("cuda").

    Returns:
      `: Input data.
      `Y`: Labels.
      `W`: (maybe empty) Weights.
    """

    X, Y, W = [], [], []
    for c, infoN in zip([X, Y, W], [data.infoX, data.infoY, data.infoW]):
        for td in infoN:
            # If the data class contains test data without 'label',
            # then, `td` is not in `subjects_batch`.
            if td in subjects_batch:
                #from IPython import embed; embed()
                if isinstance(subjects_batch[td], dict):
                    c.append( subjects_batch[td][tio.DATA].to(device) )
                else:
                    c.append( subjects_batch[td].to(device) ) # tensor
            if data.dim == "2D" and len(c) > 0:
                c[-1] = c[-1].squeeze(dim=-1)
    return (X, Y, W)

class BaseModel(torch.nn.Module):
    """
    Models inherit this class, allowing them to perform training and
    evaluation.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def initialize(self, device: str, model_state: str, log) -> None:
        """
        Initializes the model.
        Moves the operations to the selected `device`, and loads the model's
        parameters or initializes the weights/biases.

        Args:
          `device`: Device where the computations will be performed.
          `model_state`: Path to the parameters to load, or "".
          `log` (lib.utils.handlers.Log).
           changing a bit how to load the weights.
        """
        self.device = device
        self.to(self.device)

        # Load or initialize weights
        if model_state != "":
            log("Loading previous model")
            params = torch.load(model_state, map_location="cuda")
            params_k = list(params)

            for k in params_k:
                if k.startswith("last.") or "nnunet" in str(self.network.__class__).lower():
                    newk = "network." + k
                else:
                    newk = "network._model." + k
                params[newk] = params[k]
                del params[k]

            #self.load_state_dict(torch.load(model_state))
            self.load_state_dict(params, strict=False)

        else:
            # Since I'm using a torchvision model, it is already initialized
            # If I leave this, I run into problems, because it initializes
            # the convolutions various times, because first it goes to the
            # self.model and then to my custom modules, which are pointing to
            # the same convs... Now, this (initializing the same conv various
            # times) shouldn't be a problem, but it makes hard to check if
            # my modifications alter the normal functioning of the networks.
            """
            def weight_init(m):
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                    he_normal(m.weight)
                    if m.bias is not None: # for convs with bias=False
                        torch.nn.init.zeros_(m.bias)
            self.apply(weight_init)
            """
            pass

        param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log("Number of parameters: " + str(param_num))

    def fit(self, tr_loader: DataLoader, val_data: SubjectsDataset,
            iteration_start: int,
            iterations: int, val_interval: int, loss: Callable,
            val_batch_size: int,
            opt: Type[Optimizer],
            scheduler: Type[_LRScheduler],
            callbacks: List[Callable], log,
            history: dict) -> None:
        """
        Trains the NN.
        Note: I was wondering why the values in scores/results-LastEpoch.json
        and the masks in preds/ do not match (although they strongly correlate).
        The reason is that there is a pruning step in between these two.
        Therefore, "it's not a bug, it's a feature", i.e., my analysis are valid.

        Args:
          `tr_loader`: Training data.
          `val_loader`: Validaiton set.
          `iteration_start`: Iteration in which the training will start (usually 1).
          `iterations`: Iterations to train the model. If 0, no train.
          `val_interval`: After how many epochs to perform validation.
          `loss`: Loss function.
          `val_batch_size`: Batch size at validation time.
          `opt`: Optimizer.
          `scheduler`: Learning rate scheduler (for lr decay).
          `callbacks`: List of callback functions.
          `log` (lib.utils.handlers.Log).
        """
        t0 = time.time()
        outputPath = log.path.replace("log.txt", "")
        it = iteration_start
        scaler = amp.GradScaler()

        # Used by callback functions (Sauron)
        channels_history = history["channels_history"]
        val_loss_history = history["val_loss_history"]
        tr_loss_history = history["tr_loss_history"]
        if "mod_patience" in history:
            from lib.models.Sauron import DropChannels
            for mod in self.modules():
                if isinstance(mod, DropChannels):
                    mod.thr = history["mod_thr"][mod.name]
                    mod.patience = history["mod_patience"][mod.name]

        if len(val_data) > 0:
            os.makedirs(f"{outputPath}/scores")

        # Tensoboard path
        tb_path = "/".join(outputPath.split("/")[:-3]) + "/tensorboard/" + "_".join(outputPath.split("/")[-3:])[:-1]
        writer = SummaryWriter(tb_path)

        callCallbacks(callbacks, "_start_training", locals())

        tr_loss = 0 # exponential moving average (alpha=0.99)

        while it <= iterations:

            callCallbacks(callbacks, "_start_iteration", locals())

            self.train()

            for tr_i, subjects_batch in enumerate(tr_loader):
                X, Y, W = unwrap_data(subjects_batch,
                        tr_loader.dataset.dataset, self.device)
                info = subjects_batch["info"]

                callCallbacks(callbacks, "_start_train_iteration", locals())

                # NOTE: I'm not using mixed precision here because I get an
                # error when doing backpropagation: RuntimeError: Function 'CudnnConvolutionBackward' returned nan values in its 1th output.
                # It might be that a large number that fits in 32bits goes to
                # inf when converting to 16bits, or small numbers becoming 0
                # Try mixed precision when using the real dataset
                output = self(X)
                tr_loss_tmp = loss(output, Y)
                tr_loss = 0.99*tr_loss + 0.01*tr_loss_tmp.cpu().detach().numpy()

                # Optimization
                opt.zero_grad()
                tr_loss_tmp.backward()
                callCallbacks(callbacks, "_after_compute_grads", locals())
                opt.step()

                if it % val_interval == 0:
                    val_str = ""
                    if len(val_data) > 0:
                        log("Validation")
                        val_str = self.evaluate(val_data, val_batch_size,
                                os.path.join(outputPath, f"scores/results-{it}.json"),
                                loss, callbacks)
                        val_loss = float(re.match("Val loss: -?([0-9]*\.[0-9]*)", val_str)
                                .group().split(" ")[-1])
                        val_loss_history.append(val_loss)
                        # NOTE: Here, the "tr_loss" is going to behave in a different
                        # because it's computed as a moving average. I also put it here
                        # so that its len = len(val_loss_history)
                        tr_loss_history.append(tr_loss)
                        writer.add_scalar("val_loss", val_loss, it)

                    eta = datetime.fromtimestamp(time.time() + (iterations-it)*(time.time()-t0)/it).strftime("%Y-%m-%d %H:%M:%S")
                    log(f"Iteration: {it}. Loss: {tr_loss}. {val_str} ETA: {eta}")
                    writer.add_scalar("tr_loss", tr_loss, it)
                    writer.close()
                    self.train()

                callCallbacks(callbacks, "_end_train_iteration", locals())
                if scheduler:
                    scheduler.step()
                it += 1
                if it > iterations:
                    break

                #if it > 200:
                #    asdasd

            # Save utilized GPU memory
            #out = os.popen('nvidia-smi').read()
            #with open("output_nvidia-smi", "w") as f:
            #    f.write(out)
            #raise Exception("para")
            #if e > 400: # For stopping the training before 3 days (kits)
            #    break
        callCallbacks(callbacks, "_end_training", locals())


    def evaluate(self, data: SubjectsDataset, batch_size: int,
            path_scores: str, loss: Callable=None,
            callbacks: List[Callable]=[], save_predictions: bool=False) -> str:

        self.eval()

        results = {}

        val_loss = -1 if loss is None else 0

        loader = DataLoader(data, batch_size=batch_size)
        Y_true, Y_hat, ids = [], [], []

        with torch.no_grad():
            for i, subjects_batch in enumerate(loader):
                X, Y, W = unwrap_data(subjects_batch,
                        loader.dataset.dataset, self.device)
                info = subjects_batch["info"]

                output = self(X)
                if not loss is None:
                    val_loss_tmp = loss(output, Y)
                    val_loss += val_loss_tmp.cpu().detach().numpy() * (len(info) / len(data))

                y_pred_cpu = output[0].cpu().detach().numpy()
                y_true_cpu = Y[0].cpu().detach().numpy()
                #Y_true.extend(np.argmax(y_true_cpu, axis=0))
                Y_true.extend(y_true_cpu)
                Y_hat.extend(y_pred_cpu)
                ids.extend(info["id"])

                #for sub_i in range(y_pred_cpu.shape[0]):
                #    results[info["id"][sub_i]] = Measure.all(y_pred_cpu[sub_i],
                #            y_true_cpu[sub_i], info)

        Y_true = np.array(Y_true)
        Y_hat = np.array(Y_hat)

        if save_predictions:
            columns = [data.dataset.classes[k] for k in sorted(data.dataset.classes.keys())]
            dd = {columns[col_i]:Y_hat[:, col_i] for col_i in range(Y_hat.shape[1])}
            dd["ID"] = ids
            df = pd.DataFrame(columns=["ID"]+columns, data=dd).sort_values(by="ID")

            save_folder = os.path.dirname(path_scores)
            file_name = path_scores.split("/")[-1].replace("results", "predictions").replace("json", "csv")
            df.to_csv(os.path.join(save_folder, file_name), index=False)

            #from IPython import embed; embed(); asd

        # This will fail if one column of Y_true is all zeros.
        auc = roc_auc_score(Y_true, Y_hat, multi_class="ovr", average=None)
        if isinstance(auc, float):
            results = {0: auc}
            auc_text = auc
        else:
            results = {i:v for i, v in enumerate(auc)}
            auc_text = auc[data.dataset.measure_classes_mean]


        with open(path_scores, "w") as f:
            f.write(json.dumps(results))

        val_str = ""
        if not loss is None:
            val_str += f"Val loss: {val_loss}. AUC: {auc_text}."

        return val_str

