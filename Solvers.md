This is quick evaluation of different solvers performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Non-linearity: ReLU
Augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.

Joint parameters:

base_lr: 0.01
lr_policy: "step"
gamma: 0.1
momentum: 0.9
weight_decay: 0.0005

### Solvers

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| SGD with momentum |0.471| 2.36 | |
| Nesterov | **0.473 **| **2.34** |  |
| RMSProp |0.327| 3.20 | rms_decay=0.9, delta=1.0  |


Other, than lr=0.01, step=100K
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| RMSProp | 0.453| 2.45 | rms_decay=0.9, delta=1.0, base_lr: 0.045, stepsize=10K. gamma=0.94 (from [here](https://github.com/smichalowski/google_inception_v3_for_caffe)) |
| RMSProp | 0.451| 2.43| rms_decay=0.9, delta=1.0, base_lr: 0.1, stepsize=10K. gamma=0.94 |
| RMSProp | 0.472| 2.36| rms_decay=0.9, delta=1.0, base_lr: 0.1, stepsize=5K. gamma=0.94 |
| RMSProp |0.486| 2.28 | rms_decay=0.9, delta=1.0, lr=0.1, linear lr_policy  |
| SGD with momentum, linear | **0.493** | **2.24** | linear lr_policy  |


Not converge at all:

ADAM: 
lr=0.001 m=0.9 m2=0.999 delta=1e-8
lr=0.001 m=0.95 m2=0.999 delta=1e-8
lr=0.001 m=0.95 m2=0.999 delta=1e-7
lr=0.01 m=0.9 m2=0.999 delta=1e-8
lr=0.01 m=0.9 m2=0.999 delta=1e-7
lr=0.01 m=0.9 m2=0.999 delta=1e-9
lr=0.01 m=0.9 m2=0.99 delta=1e-8
lr=0.01 m=0.9 m2=0.999 delta=1e-8
lr=0.01 m=0.95 m2=0.999 delta=1e-9

AdaDelta:
delta: 1e-5

RMSProp, 
lr=0.01, rms_decay=0.5
lr=0.01, rms_decay=0.9
lr=0.01, rms_decay=0.95
lr=0.01, rms_decay=0.98
lr=0.001, rms_decay=0.9
lr=0.001, rms_decay=0.98

Converge, but much worse that SGD:
Adagrad, lr=0.01, lr=0.02
AdaDelta: delta: 1e-6, delta: 1e-7, delta: 1e-8
RMSProp, lr=0.01, rms_decay=0.99

 
The best hyperparameters are shown in graphs, the rest doesn`t even start to learn.
If anyone knows good hyperparameters for adaptive methods (funny, isn`t it?), please, tell me.

lr_policy moved to [Lr_policy.md](Lr_policy.md)

![CaffeNet128 test accuracy](/logs/solvers/img/0.png)


![CaffeNet128 test loss](/logs/solvers/img/2.png)


![CaffeNet128 train loss](/logs/solvers/img/6.png)

