This is quick evaluation of #filters impact on performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Default augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.


### Network width

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| 4x wider | **0.563** | **1.92** | Still no overfitting %) | 
| 2x wider| 0.533 | 2.04 | |
| sqrt(2)x wider | 0.506 | 2.17 | |
| Default | 0.471 | 2.36 | |
| sqrt(2)x narrower | 0.460 | 2.41 | |
| 2x narrower |  0.416 | 2.68 | |


[logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/contrib/complexity)


![CaffeNet128 test accuracy](/logs/contrib/complexity/img/0.png)


![CaffeNet128 test loss](/logs/contrib/complexity/img/2.png)


![CaffeNet128 train loss](/logs/contrib/complexity/img/6.png)


