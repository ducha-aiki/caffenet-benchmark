This is quick evaluation of different lr_policies with SGD solver performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Non-linearity: ReLU
Augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.


### LR-policy

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Step 100K |0.470| 2.36 | Default caffenet solver, max_iter=320K |
| Poly lr, p=0.5, sqrt |0.483| 2.29 | bvlc_quick_googlenet_solver, All the way worse than "step", leading at finish |
| Poly lr, p=2.0, sqr |0.483| 2.299 | |
| Poly lr, p=1.0, linear |**0.493**|***2.24*** | |
| Poly lr, p=1.0, linear |0.466|2.39 | max_iter=160K|
| Exp, 0.035 |0.441|2.53 | max_iter=160K, stepsize=2K, gamma=0.915, same as in base_dereyly|

#### LR-policy-BatchNorm-Dropout = 0.2

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Step 100K |**0.527**| **2.09** | Default caffenet solver, max_iter=320K |
| Poly lr, p=1.0, linear |0.496|2.24 | max_iter=105K,|
| Poly lr, p=1.0, start_lr=0.02 |0.505| 2.21 | max_iter=105K|
| Exp, 0.035 |0.506| 2.19 | max_iter=160K, stepsize=2K, gamma=0.915, same as in base_dereyly|


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/lr_policy), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/lr_policy)


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

"Linear_2x_faster"
max_iter: 160000

"0035_exp_160K"
max_iter: 160000
base_lr: 0.035
lr_policy: "step"
stepsize: 2000
gamma: 0.915


"Linear_3x_faster"
max_iter: 105000

See learning_rate graph

![CaffeNet128 test accuracy](/logs/lr_policy/img/0.png)


![CaffeNet128 test loss](/logs/lr_policy/img/2.png)


![CaffeNet128 lr_rate](/logs/lr_policy/img/4.png)


![CaffeNet128 train loss](/logs/lr_policy/img/6.png)

