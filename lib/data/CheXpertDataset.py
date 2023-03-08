import torch, os, random, time, re
import nibabel as nib
import pandas as pd
import numpy as np
from lib.data.BaseDataset import BaseDataset
import torchio as tio
import lib.loss as loss
from typing import List
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import InstanceNorm2d
from lib.models.UNet import UNet
from lib.models.nnUNet import nnUNet
from lib.models.Sauron import Sauron

"""
Adam, lr=1e-4 with WD
CE loss
25k iters with BS=16, or 18k with BS=32
input mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
use sigmoid for the different 5 classes
As DA: rotation, translation and scaling

"The competition requires participants to submit the trainedmodels for evaluation of the AUC score on predicting 5 selected diseases, i.e., Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion."
"""

class CheXpertDataset(BaseDataset):
    """
    """
    # Dataset info
    name = "chexpert"

    dim = "2D"
    classes = {0: 'Enlarged Cardiomediastinum', 1: 'Cardiomegaly', 2: 'Lung Opacity', 3: 'Lung Lesion', 4: 'Edema', 5: 'Consolidation', 6: 'Pneumonia', 7: 'Atelectasis', 8: 'Pneumothorax', 9: 'Pleural Effusion', 10: 'Pleural Other', 11: 'Fracture', 12: 'Support Devices', 13: 'No Finding'}
    # Specify how the final prob. values are computed: softmax or sigmoid?
    onehot = "sigmoid"
    # Which classes will be reported during validation
    # Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion
    measure_classes_mean = np.array([1, 4, 5, 7, 9])
    data_path = ""
    infoX = ["image"] # Image modalities
    infoY = ["label"] # Labels
    infoW = [] # Optional Weights

    # Sauron properties. These are automatically decided based on whether Sauron is used (see lib.utils).
    #dist_fun = ""
    #imp_fun = ""
    sf = 2

    # Optimization strategy
    opt = {
            #"architecture": {"network_name": "resnet50", "n_classes": 14,
            #"dist_fun": dist_fun, "imp_fun": imp_fun},
            #"loss": loss.CrossEntropyLoss_Distance,
            "batch_size": 32,
            "iterations": 40000,
            "optim": torch.optim.Adam,
            "optim_params": {"lr": 1e-3, "weight_decay": 1e-5},
            "scheduler": "poly",
            }

    # Data augmentation strategy

    transforms_train = tio.Compose([
        tio.transforms.RandomAffine(scales=[1, 1, 1, 1, 1, 1],
            degrees=[0, 0, 0, 0, -10, 10], translation=[0, 0, 0, 0, 0, 0], p=0.5,
            default_pad_value=0),
        tio.transforms.RandomAffine(scales=[0.9, 1.1, 0.9, 1.1, 1, 1],
            degrees=[0, 0, 0, 0, 0, 0], translation=[0, 0, 0, 0, 0, 0], p=0.5),
        tio.transforms.ZNormalization(),
        ])
    transforms_val = tio.Compose([
        tio.transforms.ZNormalization(),
        ])
    transforms_test = tio.Compose([
        tio.transforms.ZNormalization(),
        ])

    def __init__(self, fold: int, percentage: float, resolution: int):
        """
        Divide the data into train/validation splits.

        Args:
          `fold`: Test fold. Training folds are the others.
          `percentage`: Percentage of training data used (the test is the same).
          `resolution`: Resolution of the images. # 224 or 448
        """

        self.transforms_dict = {"train": self.transforms_train,
                "validation": self.transforms_val, "test": self.transforms_test}

        self.subjects_dict = {"train": [], "validation": [], "test": []}

        #from IPython import embed; embed(); asd
        df = pd.read_csv(os.path.join(self.data_path,
            "fivefoldsplits", f"fold{fold}_keep1.0.csv"))
        df['Path'] = df['Path'].str.replace(
                "CheXpert-v1.0/train",
                os.path.join(self.data_path, f"CheXpert_{resolution}_Train"),
                regex=False)
        # Note that the test set is always the same
        df_test = df[df["split"]=="test"]

        if percentage == 1:
            df_train = df[df["split"]=="train"]
            df_val = df[df["split"]=="validation"]
        else:
            df2 = pd.read_csv(os.path.join(self.data_path,
                "fivefoldsplits", f"fold{fold}_keep{percentage}.csv"))
            df2['Path'] = df2['Path'].str.replace(
                    "CheXpert-v1.0/train",
                    os.path.join(self.data_path, f"CheXpert_{resolution}_Train"),
                    regex=False)
            df_train = df2[df2["split"]=="train"]
            df_val = df2[df2["split"]=="validation"]

        #df_train = df_train.sample(frac=1) # Randomize


        ### Smaller datasets to debug
        #df_val = df_train.iloc[:100]
        #df_test = df_train.iloc[:112]
        #df_train = df_train.iloc[:100]


        ids_train = df_train['Path'].str.replace(self.data_path, "", regex=False).str.replace("/", "_", regex=False).str.replace(".jpg", "", regex=False)
        ids_val = df_val['Path'].str.replace(self.data_path, "", regex=False).str.replace("/", "_", regex=False).str.replace(".jpg", "", regex=False)
        ids_test = df_test['Path'].str.replace(self.data_path, "", regex=False).str.replace("/", "_", regex=False).str.replace(".jpg", "", regex=False)

        paths_train = df_train['Path']
        paths_val = df_val['Path']
        paths_test = df_test['Path']

        columns = [self.classes[k] for k in sorted(self.classes.keys())]
        true_train = np.array(df_train[columns])
        true_val = np.array(df_val[columns])
        true_test = np.array(df_test[columns])

        for ids, paths, true_vectors, whichset in zip([ids_train, ids_val, ids_test], [paths_train, paths_val, paths_test], [true_train, true_val, true_test], ["train", "validation", "test"]):
            for id_im, path, true in zip(ids, paths, true_vectors):
                self.subjects_dict[whichset].append(tio.Subject(
                    image=tio.ScalarImage(path),
                    label=true,
                    info={
                        "id": id_im, # Remove the first /
                        "patch_size": (resolution, resolution, 1)
                        }
                    ))

        print("Training images", len(self.subjects_dict["train"]))
        print("Validation images", len(self.subjects_dict["validation"]))
        print("Test images", len(self.subjects_dict["test"]))

