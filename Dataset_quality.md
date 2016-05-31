This is quick evaluation of trainig set quality impact on performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Default augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.



### Dataset quality

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default, clean labels | **0.471** | **2.36** | |
| 5% incorrect labels |0.458| 2.45 | |
| 10% incorrect labels | 0.447| 2.58 | |



[logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/contrib)


![CaffeNet128 test accuracy](/logs/contrib/img/0_dataset_q.png)


![CaffeNet128 test loss](/logs/contrib/img/2_dataset_q.png)


![CaffeNet128 train loss](/logs/contrib/img/6_dataset_q.png)

