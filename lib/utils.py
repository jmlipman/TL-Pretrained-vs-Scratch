import argparse, inspect, os, sys
from lib.data.BaseDataset import BaseDataset
from typing import Type
from torch.nn.parameter import Parameter as TorchParameter
import torch, json
from lib.paths import data_path
import numpy as np
import types, random, time, pickle
from datetime import datetime
from torch import Tensor
import torchio as tio
from torch.utils.data import DataLoader
import nibabel as nib
from typing import List, Tuple, Union
import pandas as pd
from sklearn.cluster import KMeans
import lib.loss as loss

valid_network_names = ["resnet50", "resnet50_25", "resnet50_50",
        "resnet50_ext1", "nnunet_encoder_v2"]

def getPCname() -> str:
    pc_name = os.uname()[1]
    return pc_name


def getDataset(dataset: str) -> Type[BaseDataset]:
    """
    Retrieves the dataset given its Dataset.name (files in lib/data/)

    Args:
      `dataset`: name of the dataset.

    Returns:
      Dataset object.
    """
    # Datasets
    # Placing the imports here avoid circular import
    if dataset == "chexpert":
        from lib.data.CheXpertDataset import CheXpertDataset
        return CheXpertDataset
    elif dataset == "isic2020":
        from lib.data.ISIC2020Dataset import ISIC2020Dataset
        return ISIC2020Dataset

    raise ValueError(f"Dataset `{dataset}` not found.")


def parseArguments() -> None:
    """
    Parses, verifies, and sanitizes the arguments provided by the user.
    """

    parser = argparse.ArgumentParser(description="UNet strikes back! parser")

    parser.add_argument("--exp_name", help="Name of the experiment",
            default="baseline")

    # Data and model
    parser.add_argument("--data", help="Name of the dataset", required=True)
    parser.add_argument("--fold", help="XVal fold", required=True)
    parser.add_argument("--device", help="Pytorch device", default="cuda")
    parser.add_argument("--model_state", help="Pretrained model", default="")
    parser.add_argument("--in_filters", help="File containing in_filters", default="")
    parser.add_argument("--out_filters", help="File containing out_filters", default="")
    parser.add_argument("--resolution", help="Fold number", required=True)
    parser.add_argument("--percentage", help="Fold number", required=True)

    # Training strategy
    parser.add_argument("--network_name", help="CNN", required=True)
    parser.add_argument("--pretrained", help="Whether the CNN is pretrained", required=True)
    parser.add_argument("--sauron", help="Whether to prune with Sauron", required=True)

    # epochs_start can be useful to continue the training when using lr_decay
    parser.add_argument("--iteration_start", help="Number of epochs", default=1)
    parser.add_argument("--iterations", help="Number of epochs", default="")
    parser.add_argument("--batch_size", help="Batch size", default="")
    parser.add_argument("--val_interval", help=f"Frequency in which validation"
            " will be computed", default=2)
    parser.add_argument("--optim", help="Optimizer (e.g., adam)", default="")
    parser.add_argument("--lr", help="Learning rate", default="")
    parser.add_argument("--wd", help="Weight decay", default="")
    parser.add_argument("--momentum", help="Momentum (use with SGD)", default="")
    parser.add_argument("--nesterov", help="Momentum (use with SGD)", default="")

    # Other
    parser.add_argument("--seed_train", help="Random seed for pytorch, np, random",
            default="42")
    parser.add_argument("--history", help="Location of val_loss_history, etc. Useful for loading Sauron's state.", default="")

    args = parser.parse_args()

    cfg = {"data": getDataset(args.data)}

    # OPTIMIZER
    # If the user specifies the optimizer, it should also specify other
    # optimizer-related params, such as lr.
    optim_name = args.optim.lower()
    # Use default config. specific to the dataset
    if optim_name == "":
        cfg["optim"] = cfg["data"].opt["optim"]
    elif optim_name  == "adam":
        cfg["optim"] = torch.optim.Adam
    elif optim_name  == "sgd":
        cfg["optim"] = torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer `{args.optim}`"
                f" only 'adam' and 'sgd' available at the moment")

    # Grab the default opt params, and override those provided by the user
    cfg["optim_params"] = cfg["data"].opt["optim_params"]
    if not args.fold in ["1", "2", "3", "4", "5"]:
        raise ValueError(f"--fold must be between 1 and 5; not {args.fold}")
    cfg["fold"] = args.fold

    if args.lr != "":
        # If lr is not a number -> ValueError
        if args.pretrained == "1":
            cfg["optim_params"]["lr"] = float(args.lr)/10
        else:
            cfg["optim_params"]["lr"] = float(args.lr)
    if args.wd != "":
        cfg["optim_params"]["weight_decay"] = float(args.wd)

    if cfg["optim"] is torch.optim.SGD:
        if args.momentum != "":
            cfg["optim_params"]["momentum"] = float(args.momentum)
        if args.nesterov != "":
            cfg["optim_params"]["nesterov"] = bool(args.nesterov)

    # LOSS FUNCTION
    if not args.sauron in ["0", "1"]:
        raise ValueError("--sauron must be either 1 or 0")
    if args.sauron == "1":
        cfg["loss"] = loss.CrossEntropyLoss_Distance
    else:
        cfg["loss"] = loss.CrossEntropyLoss

    # SCHEDULER (LR DECAY)
    if "scheduler" in cfg["data"].opt:
        allowed_schedulers = ["poly"]
        if cfg["data"].opt["scheduler"] in allowed_schedulers:
            cfg["scheduler"] = cfg["data"].opt["scheduler"]
        else:
            raise Exception(f"Unknown scheduler: f{cfg['data'].opt['scheduler']}")
    else:
        cfg["scheduler"] = None


    # ARCHITECTURE
    if not args.network_name in valid_network_names:
        raise ValueError(f"--network_name `{args.network_name}` unknown. "
                f"Valid network names: {str(valid_network_names)}")
    if not args.pretrained in ["0", "1"]:
        raise ValueError("--pretrained must be either 1 or 0")
    cfg["architecture"] = {"network_name": args.network_name,
            "n_classes": len(cfg["data"].classes),
            "pretrained": bool(int(args.pretrained))}
    if args.sauron == "1":
        if args.network_name == "mobilenetv2":
            raise ValueError("Sauron cannot work with mobilenetv2")
        cfg["architecture"]["dist_fun"] = "euc_norm"
        cfg["architecture"]["imp_fun"] = "euc_rand"
    else:
        cfg["architecture"]["dist_fun"] = ""
        cfg["architecture"]["imp_fun"] = ""

    # OTHER
    cfg["percentage"] = float(args.percentage)
    cfg["resolution"] = int(args.resolution)
    #if not cfg["percentage"] in np.arange(0.1, 1.1, 0.1):
    #    raise ValueError(f"--percentage must be in [0.1, 0.2, .., 1.0]")

    if not cfg["resolution"] in [224, 448, 672, 896]:
        raise ValueError("--resolution must be either 224, 448, 672, or 896")

    pc_name = getPCname()
    cfg["data"].data_path = data_path[args.data][pc_name]
    #cfg["architecture"] = cfg["data"].opt["architecture"]
    if args.iterations!= "":
        cfg["iterations"] = int(args.iterations)
    else:
        cfg["iterations"] = cfg["data"].opt["iterations"]

    if args.batch_size != "":
        cfg["batch_size"] = int(args.batch_size)
    else:
        cfg["batch_size"] = cfg["data"].opt["batch_size"]

    cfg["val_interval"] = int(args.val_interval)
    cfg["iteration_start"] = int(args.iteration_start)

    # This "history" is only used with Sauron
    cfg["history"] = {}
    if args.history != "" and os.path.isdir(args.history):
        cfg["history"]["path"] = args.history
        with open(os.path.join(args.history, "val_loss_history.pkl"), "rb") as f:
            cfg["history"]["val_loss_history"] = pickle.load(f)
        with open(os.path.join(args.history, "tr_loss_history.pkl"), "rb") as f:
            cfg["history"]["tr_loss_history"] = pickle.load(f)
        with open(os.path.join(args.history, "channels_history.pkl"), "rb") as f:
            cfg["history"]["channels_history"] = pickle.load(f)
        with open(os.path.join(args.history, "mod_thr.pkl"), "rb") as f:
            cfg["history"]["mod_thr"] = pickle.load(f)
        with open(os.path.join(args.history, "mod_patience.pkl"), "rb") as f:
            cfg["history"]["mod_patience"] = pickle.load(f)

        cfg["architecture"]["filters"] = {}
        df_in = pd.read_csv(os.path.join(args.history, "in_filters"), sep="\t")
        df_out = pd.read_csv(os.path.join(args.history, "out_filters"), sep="\t")
        cfg["architecture"]["filters"]["in"] = {col_name: df_in[col_name].iloc[-1] for col_name in df_in.columns}
        cfg["architecture"]["filters"]["out"] = {col_name: df_out[col_name].iloc[-1] for col_name in df_out.columns}
    else:
        cfg["history"]["channels_history"] = {}
        cfg["history"]["val_loss_history"] = []
        cfg["history"]["tr_loss_history"] = []
        cfg["architecture"]["filters"] = {}

    # For some reason, I need to import this here. Otherwise I would need to
    # use 'global', which I will avoid at all costs
    from lib.paths import output_path
    if os.path.isdir(output_path[pc_name]):
        cfg["path"] = output_path[pc_name]
    else:
        raise ValueError(f"Output path `{output_path[pc_name]}` set in 'lib/paths.py'"
                f" is not a folder")

    if args.model_state != "":
        if not os.path.isfile(args.model_state):
            raise ValueError(f"The pretrained model specified in --model_state"
                    f" `{args.model_state}` does not exist.")
    cfg["model_state"] = args.model_state

    if args.device in ["cuda", "cpu"]:
        cfg["device"] = args.device
    else:
        raise ValueError(f"Unknown device `{args.device}`."
                          " Valid options: cuda, cpu")

    if args.seed_train == "":
        seed_train = int(str(int(np.random.random()*1000)) + str(time.time()).split(".")[-1][:4])
    else:
        seed_train = int(args.seed_train)

    torch.manual_seed(seed_train)
    torch.cuda.manual_seed(seed_train)
    np.random.seed(seed_train)
    random.seed(seed_train)
    cfg["seed_train"] = seed_train

    return cfg


class Log:
    """
    Prints and stores the output of the experiments.
    """
    def __init__(self, path: str):
        """
        Args:
          `path`: Log file path.
        """
        self.path = path
        #self.log_path = path_handler.join("log.txt")
        #self.config_path = path_handler.join("config.json")

    def __call__(self, text: str, verbose: bool=True):
        """
        Saves the text into a log file.

        Args:
          `text`: Text to log.
          `verbose`: Whether to print `text` (0/1).
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{now}: {text}"
        if self.path != "": # Empty path -> disables log
            with open(self.path, "a") as f:
                f.write(text + "\n")

        if verbose:
            print(text)

    def saveConfig(self, cfg: dict) -> None:
        """
        Saves configuration dictionary `cfg` in a config.json file.

        Args:
          `cfg`: Configuration dictionary

        """

        def _serialize(obj: object):
            """Serializes data to be able to utilize json format.
            """
            if isinstance(obj, (int, float, str)):
                return obj

            elif isinstance(obj, (list, tuple)):
                return [_serialize(o) for o in obj]

            elif isinstance(obj, dict):
                newobj = {}
                for k in obj:
                    if not isinstance(k, (str, int, float, bool)):
                        newobj[k.__class__.__name__] = _serialize(obj[k])
                    else:
                        newobj[k] = _serialize(obj[k])
                    #from IPython import embed; embed()
                return newobj

            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            elif isinstance(obj, types.FunctionType):
                # Loss functions in lib.losses.py
                return obj.__name__

            elif isinstance(obj, (type)):
                if obj.__module__.startswith("torch.optim."):
                    # Optimizers
                    return obj.__name__

                # This includes data in lib.data
                newobj = {}
                attributes = inspect.getmembers(obj, lambda x:not(inspect.isroutine(x)))
                attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
                for name, att in attributes:
                    newobj[name] = _serialize(att)
                return newobj

            elif isinstance(obj, type(None)):
                return "None"

            elif hasattr(obj, "__module__"):
                if obj.__module__.startswith("lib.models."):
                    # Models
                    newobj = {}
                    newobj["model"] = obj.__module__
                    for att in obj.params:
                        newobj[att] = _serialize(getattr(obj, att))
                    return newobj

                elif obj.__module__.startswith("torchio.transforms"):
                    newobj = {}
                    attributes = inspect.getmembers(obj, lambda x:not(inspect.isroutine(x)))
                    attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
                    for name, att in attributes:
                        newobj[name] = _serialize(att)
                    return newobj
                else:
                    print(f"Warning: The object `{obj}` of type {type(obj)} might"
                           " not have been logged properly")
                    return str(type(obj))

            else:
                print(f"Warning: The object `{obj}` of type {type(obj)} might"
                       " not have been logged properly")
                return str(type(obj))

        serialized_cfg = _serialize(cfg)
        del serialized_cfg["data"]["opt"] # To avoid logging duplicate info

        with open(self.path, "w") as f:
            f.write(json.dumps(serialized_cfg))

        print("\n### SAVED CONFIGURATION ###\n")
        print(serialized_cfg)

def he_normal(w: TorchParameter):
    """
    He normal initialization.

    Args:
      `w` (torch.Tensor): Weights.

    Returns:
      Normal distribution following He initialization.
    """

    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    return torch.nn.init.normal_(w, 0, np.sqrt(2/fan_in))


def scaleHalfGroundTruth(y_true: Tensor) -> Tensor:
    """
    Used for Deep Supervision. It halfs the size (H,W,D) of the ground truth.

    Args:
      `y_true`: Tensor containing the ground truth.

    Returns:
      Tensor with half resolution as `y_true`.
    """
    dd = [torch.linspace(-1, 1, i//2) for i in y_true.shape[2:]]
    mesh = torch.meshgrid(dd)
    grid = torch.stack(mesh, -1).cuda()
    grid = torch.stack([grid for _ in range(y_true.shape[0])])
    try:
        resized = torch.nn.functional.grid_sample(y_true, grid, mode="nearest")
    except:
        from IPython import embed; embed()
        raise Exception("para")
        pass
    return resized

def softmax2onehot(image: np.array) -> np.array:
    """
    Convert a softmax probability matrix into a onehot-encoded matrix.

    Args:
      `image` (np.array): CHWD

    Returns:
      One-hot encoded matrix.
    """
    result = np.zeros_like(image)
    labels = np.argmax(image, axis=0)
    for i in range(image.shape[0]):
        result[i] = labels==i
    return result

def sigmoid2onehot(image: np.array) -> np.array:
    """
    Convert a sigmoid probability matrix into a onehot-encoded matrix.
    The difference with softmax prob. matrices is that sigmoid allows
    labels to overlap, i.e., pixels can have multiple labels.

    Args:
      `image` (np.array): CHWD

    Returns:
      One-hot encoded matrix.
    """
    thr = 0.5
    result = 1.0*(image > thr)
    return result

def resample(image_path: str="", label_path: str="",
        voxres: Tuple[float]=(), size: List[int]=[]) -> Union[List[nib.Nifti1Image],
                                                      nib.Nifti1Image]:
    """
    Resamples an image (and its label if provided) into a specific voxel
    resolution or image size. This function is used for pre- and postprocessing
    and it can be used in two different ways:
     - Option 1: Give image_path, label_path and voxres (resample).
     - Option 2: Give label_path and size (resize).
    The raison d'être of this function is to provide with a single interface
    for resampling and resizing, which became necessary as 'resampling back to
    the original space' did not yield the same image size as the original
    images. Thus, for preprocessing, resampling is used, and, for
    postprocessing, resizing is used.

    Why I'm passing the path instead of the image? TorchIO.

    Args:
      `image_path`: Location of the image to be resampled/resized.
      `label_path`: Location of the ground truth or prediction.
      `voxres`: Voxel resolution. If len != 3, torchio might complain.
      `size`: Image dimensions. If len != 3, torchio might complain.

    Returns:
      Either a list with the image and its ground truth resampled, or
      the prediction resized.

    """
    raise Exception("Deprecated. Use 'resamplev2'")

    if label_path != "" and not os.path.isfile(label_path):
        raise ValueError(f"label_path `{label_path}` does not exist.")
    if image_path != "" and not os.path.isfile(image_path):
        raise ValueError(f"image_path `{image_path}` does not exist.")

    params = {}
    if image_path != "":
        params["im"] = tio.ScalarImage(image_path)
        print(params["im"])
    if label_path != "":
        params["label"] = tio.LabelMap(label_path)

    subject = [tio.Subject(**params)]
    if len(voxres) > 0:
        trans = tio.Resample(voxres)
    else:
        trans = tio.transforms.Resize(size, image_interpolation="nearest")
        if image_path != "":
            raise ValueError("Resize (with 'size') can only be used for labels")

    transforms = tio.Compose([trans])
    sd = tio.SubjectsDataset(subject, transform=transforms)
    loader = DataLoader(sd, batch_size=1, num_workers=4)

    results = []
    sub = list(loader)[0]
    if image_path != "":
        im = sub["im"]["data"][0,0].detach().cpu().numpy()
        aff = sub["im"]["affine"][0].detach().cpu().numpy()
        results.append( nib.Nifti1Image(im, affine=aff) )

    if label_path != "":
        seg = sub["label"]["data"][0,0].detach().cpu().numpy()
        seg_aff = sub["label"]["affine"][0].detach().cpu().numpy()
        results.append( nib.Nifti1Image(seg, affine=seg_aff) )

    if len(results) == 1:
        return results[0]
    return results

def resamplev2(images: Union[List[tio.Image], tio.Image],
        voxres: Tuple[float]=(),
        size: List[int]=[]) -> Union[List[nib.Nifti1Image], nib.Nifti1Image]:
    """
    Resamples an image (and its label if provided) into a specific voxel
    resolution or image size. This function is used for pre- and postprocessing
    and it can be used in two different ways:
     - Option 1: Give image_path, label_path and voxres (resample).
     - Option 2: Give label_path and size (resize).
    The raison d'être of this function is to provide with a single interface
    for resampling and resizing, which became necessary as 'resampling back to
    the original space' did not yield the same image size as the original
    images. Thus, for preprocessing, resampling is used, and, for
    postprocessing, resizing is used.

    Why I'm passing the path instead of the image? TorchIO.

    Args:
      `image_path`: Location of the image to be resampled/resized.
      `label_path`: Location of the ground truth or prediction.
      `voxres`: Voxel resolution. If len != 3, torchio might complain.
      `size`: Image dimensions. If len != 3, torchio might complain.

    Returns:
      Either a list with the image and its ground truth resampled, or
      the prediction resized.

    """

    if len(voxres) == len(size) == 0:
        raise Exception("Either 'voxres' or 'size' must be indicated")
    if len(voxres) != 0 and len(size) != 0:
        raise Exception("Either 'voxres' or 'size' must be indicated (not both)")
    if len(voxres) == 0 and len(size) != 3:
        raise Exception("'size' should have only 3 elements")
    if len(size) == 0 and len(voxres) != 3:
        raise Exception("'voxres' should have only 3 elements")

    if not isinstance(images, list):
        images = [images]

    for im in images:
        if not isinstance(im, tio.Image):
            raise Exception("'images' expected to be tio.Images...")

    params = {}
    scalars, labels = 0, 0
    for im in images:
        if isinstance(im, tio.ScalarImage):
            scalars += 1
            params[f"im_{scalars}"] = im
        elif isinstance(im, tio.LabelMap):
            labels += 1
            params[f"label_{labels}"] = im

    subject = [tio.Subject(**params)]
    if len(voxres) > 0:
        # This typically happens in the preprocessing, when we have images and labels
        trans = tio.Resample(voxres)
    else:
        # Here, we typically have a single image for postprocessing.
        if len(images) > 1:
            print("WARNING: We are going to use interpolation=linear. "
                  "If there are LabelMaps, I think that they are interpolated "
                  "automatically with 'nearest'. Check!")
            #trans = tio.transforms.Resize(size, image_interpolation="linear")
        trans = tio.transforms.Resize(size, image_interpolation="linear")

    transforms = tio.Compose([trans])
    sd = tio.SubjectsDataset(subject, transform=transforms)
    loader = DataLoader(sd, batch_size=1, num_workers=4)

    results = []
    sub = list(loader)[0]
    for i in range(1, scalars+1):
        im = sub[f"im_{i}"]["data"][0].detach().cpu().numpy()
        aff = sub[f"im_{i}"]["affine"][0].detach().cpu().numpy()
        if im.shape[0] == 1:
            im = im[0]
        results.append( nib.Nifti1Image(im, affine=aff) )

    for i in range(1, labels+1):
        seg = sub[f"label_{i}"]["data"][0,0].detach().cpu().numpy()
        seg_aff = sub[f"label_{i}"]["affine"][0].detach().cpu().numpy()
        results.append( nib.Nifti1Image(seg, affine=seg_aff) )

    if len(results) == 1:
        return results[0]
    return results

