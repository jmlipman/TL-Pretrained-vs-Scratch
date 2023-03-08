import argparse, os, importlib, inspect, json, time
from lib.metric import Metric
import nibabel as nib
import numpy as np
from lib.utils import getPCname, Log, valid_network_names
from lib.paths import data_path
from lib.models.Sauron import Sauron
import pandas as pd

# Input can be the folder of the train or test set
# It is automatically inferred which one is it.
t0 = time.time()
parser = argparse.ArgumentParser(description="Parser for evaluating data")
parser.add_argument("--data", help="Name of the dataset", required=True)
parser.add_argument("--fold", help="Fold number", required=True)
parser.add_argument("--resolution", help="Fold number", required=True)
parser.add_argument("--model_state", help="Model's parameters", required=True)
parser.add_argument("--network_name", help="CNN name", required=True)
parser.add_argument("--output", help="Output file", required=True)
parser.add_argument("--in_filters", help="File containing in_filters", default="")
parser.add_argument("--out_filters", help="File containing out_filters", default="")
parser.add_argument("--line_filters", help="In which line, in --in_filters/out_filters, can I find the number of filters. This is useful when running this script in a loop.", default="")

args = parser.parse_args()
data = args.data
resolution = int(args.resolution)
fold = args.fold
network_name = args.network_name
outputFile = args.output
modelState = args.model_state
inFilters = args.in_filters
outFilters = args.out_filters
lineFilters = args.line_filters

available_datasets = {}
pythonFiles = [x.replace(".py", "") for x in os.listdir("lib/data") if x.endswith(".py")]
for pyfi in pythonFiles:
    for name, cl in inspect.getmembers(importlib.import_module(f"lib.data.{pyfi}")):
        if inspect.isclass(cl):
            if hasattr(cl, "name"):
                available_datasets[getattr(cl, "name")] = cl

if not data in available_datasets:
    raise ValueError(f"--data `{data}` is invalid. Available datasets:"
            f" {available_datasets}")

if not os.path.isfile(modelState):
    raise ValueError(f"--model_state `{modelState}` does not exist.")

if not fold in [str(i) for i in range(1, 6)]:
    raise ValueError(f"--fold must be [1-5]")

if not resolution in [224, 448, 672, 896]:
    raise ValueError("--resolution must be either 224, 448, 672, or 896")

if not args.network_name in valid_network_names:
    raise ValueError(f"--network_name `{args.network_name}` unknown. "
            f"Valid network names: {str(valid_network_names)}")

if os.path.isfile(outputFile):
    raise ValueError(f"--output file `{outputFile}` already exists.")

folder = os.path.dirname(outputFile)
if not os.path.isdir(folder):
    os.makedirs(folder)

if os.path.isfile(inFilters) and os.path.isfile(outFilters):
    df_in = pd.read_csv(inFilters, sep="\t")
    df_out = pd.read_csv(outFilters, sep="\t")
    filters = {}
    if lineFilters:
        lineFilters = int(lineFilters)
    else:
        lineFilters = -1 # Last line
    if lineFilters >= df_in.shape[0]:
        raise ValueError("--line_filters cannot be greater than the size of the dataframe")
    filters["in"] = {col_name: df_in[col_name].iloc[lineFilters] for col_name in df_in.columns}
    filters["out"] = {col_name: df_out[col_name].iloc[lineFilters] for col_name in df_out.columns}
else:
    filters = {}

C = available_datasets[data]
pc_name = getPCname()
C.data_path =  data_path[args.data][pc_name]

data = C(fold, 0.2, resolution) # the percentage for the evaluation part is irrelevant

model = Sauron(
        network_name=args.network_name,
        n_classes=len(C.classes),
        pretrained=False,
        dist_fun="",
        imp_fun="",
        sf=2,
        filters=filters,
        )
model.initialize(device="cuda", model_state=modelState, log=Log(""))

model.evaluate(data=data.get("test"),
        batch_size=16,
        path_scores=outputFile,
        save_predictions=True)

print(f"Total time: {np.round((time.time()-t0)/60, 3)} mins.")
#from IPython import embed; embed(); asd
