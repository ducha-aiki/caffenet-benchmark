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

lr_policy moved to [Lr_policy.md](Lr_policy.md)

![CaffeNet128 test accuracy](/logs/solvers/img/0.png)


![CaffeNet128 test loss](/logs/solvers/img/2.png)


![CaffeNet128 train loss](/logs/solvers/img/6.png)

