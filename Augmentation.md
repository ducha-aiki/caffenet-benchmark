This is quick evaluation of different ways of data augmentation performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Default augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.
Additional augmentations:
- dropout of input data, dropout_ratio = 0.1
- multiscale:


### Train augmentation

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default |**0.471**| **2.36** | Random flip, random crop 128x128 from 144xN, N > 144|
| Drop 0.1 |0.306| 3.56 | + Input dropout 10%. not finished, 186K iters result |
| Multiscale |0.462| 2.40 | Random flip, random crop 128x128 from ( 144xN, - 50%, 188xN - 20%, 256xN - 20%, 130xN - 10%)

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/augmentation), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/augmentation)


![CaffeNet128 test accuracy](/logs/augmentation/img/0.png)


![CaffeNet128 test loss](/logs/augmentation/img/2.png)


![CaffeNet128 train loss](/logs/augmentation/img/6.png)

