# TL-Pretrained-vs-Scratch

Repository of the paper: Fine-tuning ImageNet-pretrained models or training from scratch for medical image classification? Analysis of multiple disentangled factors challenging previous findings


Example of use:

```
data='chexpert' # chexpert or isic2020
fold=1 # 1, 2, 3, 4, 5
network_name='resnet50' # resnet50, resnet50_50, resnet50_25, resnet50_ext1, nnunet_encoder_v2
weights='scratch' # scratch or pretrained
res=224 # 224, 448, 678, 896
per=1.0 # 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0

python train.py --data $data --val_interval 20 --fold $fold --network_name $network --pretrained $weights --resolution $res --percentage $per --batch_size 16
```

Evaluation

```
python evaluate.py --data $data --fold $fold --network_name $network --output "$outputFolder"'results/results-'$iter'.json' --resolution $res --model_state "$outputFolder"'models/model-40000'
```
