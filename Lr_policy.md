This is quick evaluation of different lr_policies with SGD solver performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Non-linearity: ReLU
Augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.

Step100K == reference caffenet solver.
base_lr: 0.01
lr_policy: "step"
gamma: 0.1
momentum: 0.9
weight_decay: 0.0005

"Poly_sqrt" == bvlc_googlenet_quick_solver
lr_policy: "poly"
power: 0.5

"Poly_sqr"
lr_policy: "poly"
power: 2.0

"Linear"
lr_policy: "poly"
power: 1.0

See learning_rate graph

![CaffeNet128 test accuracy](/logs/lr_policy/img/0.png)


![CaffeNet128 test loss](/logs/lr_policy/img/2.png)


![CaffeNet128 lr_rate](/logs/lr_policy/img/4.png)


![CaffeNet128 train loss](/logs/lr_policy/img/6.png)

