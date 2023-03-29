# AnchorFormer 

# Environment

These model are implemented in PyTorch (1.8.1+cu102) version. 

The version of operation system in docker is Ubuntu 18.04.6 LTS.

The python version is 3.8.13. 

The GPU is NVIDIA Tesla V100 (16GB) and the CUDA version is CUDA 10.2.

# Note
This repository includes the pure code and configuration files of AnchorFormer model in the paper. 

# The structure mentioned in the paper
```
Feature Extraction Module: ./models/EdgeConv.py line 8-107

Transformer Encoder Block: ./models/Transformer.py line 161-208

Point Scattering Module: ./models/Transformer.py line 334-337

Transformer Decoder Block: ./models/Transformer.py line 210-239

Point Morphing Module: ./models/Morphing.py line 147-184
```

# Usage
## Install
1. Requirements
```
argparse
easydict
h5py
matplotlib
numpy
open3d==0.9
opencv-python
pyyaml
scipy
tensorboardX
timm==0.4.5 
tqdm
transforms3d
```
2. c++ extensions
```
bash ./extensions/install.sh
```
3. Standard PointNet++ lib 
(ref to "https://github.com/erikwijmans/Pointnet2_PyTorch")



## train
```
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --master_port=${PORT} --nproc_per_node=${NGPUS} main.py --launcher pytorch --sync_bn ${PY_ARGS}
```
example:

1. train from start
```
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python -m torch.distributed.launch --node_rank=0 --nnodes=1 --master_port=13232 --nproc_per_node=4 main.py --launcher pytorch --sync_bn --config ./cfgs/PCN_models/AnchorFormer.yaml --exp_name try_to_train_anchorformer (--val_freq 10 --val_interval 50) 
```

2. resume from last break-point
```
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python -m torch.distributed.launch --node_rank=0 --nnodes=1 --master_port=13232 --nproc_per_node=4 main.py --launcher pytorch --sync_bn --config ./cfgs/PCN_models/AnchorFormer.yaml --exp_name try_to_train_anchorformer --resume
```
3. resume from specified break-point 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python -m torch.distributed.launch --node_rank=0 --nnodes=1 --master_port=13232 --nproc_per_node=4 main.py --launcher pytorch --sync_bn --config ./cfgs/PCN_models/AnchorFormer.yaml --exp_name try_to_train_anchorformer --start_ckpts {CKPT_PATH}.pth
```
## test
```
CUDA_VISIBLE_DEVICES=${GPUS} python main.py --test ${PY_ARGS}
```
example:
```
CUDA_VISIBLE_DEVICES=0 python main --ckpts {CKPT_PATH}.pth --config ./cfgs/PCN_models/AnchorFormer.yaml --exp_name test_ckpt
```
