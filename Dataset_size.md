This is quick evaluation of trainig set size impact on performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Default augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.



### Dataset size

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default, 1.2M images | **0.471** | **2.36** | |
| 800K images |0.438| 2.54 | |
| 600K images |0.425| 2.63 | |
| 400K images |0.393| 2.92 | |
| 200K images |0.305| 4.04 | |



### Dataset size, no RGB scaling 
Or why input var=1 for LSUV is so important

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| 800K images |0.438| 2.54 | |
| 800K images, no scale |0.379| 2.92 | |
| 600K images |0.425| 2.63 | |
| 600K images, no scale |0.379| 2.92 | |
| 400K images |0.393| 2.92 | |
| 400K images, no scale |0.357| 3.10 | |
| 200K images |0.305| 4.04 | |
| 200K images, no scale |0.277| 4.06 | |

[logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/contrib)


![CaffeNet128 test accuracy](/logs/contrib/img/0_dataset.png)


![CaffeNet128 test loss](/logs/contrib/img/2_dataset.png)


![CaffeNet128 train loss](/logs/contrib/img/6_dataset.png)

