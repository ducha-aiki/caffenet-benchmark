This is quick evaluation of BatchNorm layer (https://github.com/BVLC/caffe/pull/3229) performance on ImageNet-2012. 

Other on-going evaluations:
- [activations](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Activations.md)
- [architectures] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Architectures.md)
- [augmentation] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Augmentation.md)

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)

Because LRN layers add nothing to accuracy, they were removed for speed reasons in further experiments.


## Batch normalization
[BN-paper](http://arxiv.org/abs/1502.03167), [caffe-PR](https://github.com/BVLC/caffe/pull/3229)
Note, that results are obtained without mentioned in paper y=kx+b additional layer.

### BN -- before or after ReLU?
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Before |0.474| 2.35 | As in paper|
| Before + scale&bias layer |0.478| 2.33 | As in paper|
| After |**0.499**| **2.21** | |
| After + scale&bias layer |0.493| 2.24 | |

So in all next experiments, BN is put after non-linearity

### BN and activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| ReLU |0.499| 2.21 | |
| RReLU |0.500| 2.20 | |
| PReLU |**0.503**| **2.19** | |
| ELU |0.498| 2.23 | |
| Maxout |0.487| 2.28| |
| Sigmoid |0.475| 2.35| |
| TanH |0.448| 2.50 | |
| No |0.384| 2.96 | |


### BN and dropout

ReLU non-linearity, fc6 and fc7 layer only

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
|  Dropout = 0.5 |0.499| 2.21 |  |
|  Dropout = 0.2 |**0.527**| **2.09** |  |
|  Dropout = 0 |0.513| 2.19 | |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs)


### BN-arch-init
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Caffenet |0.471| 2.36 ||
| Caffenet BN Before + scale&bias layer LSUV|0.478| 2.33 | |
| Caffenet BN Before + scale&bias layer Ortho|0.482| 2.31 | |
| Caffenet BN After LSUV | 0.499 | 2.21 | |
| Caffenet BN After Ortho | **0.500** | **2.20** | |

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [GoogLeNet128](http://arxiv.org/abs/1409.4842) | **0.619** | **1.61** ||
| GoogLeNet BN Before + scale&bias layer LSUV| 0.603 | 1.68  | |
| GoogLeNet BN Before + scale&bias layer Ortho| 0.607| 1.67 | |
| GoogLeNet BN After LSUV |0.596| 1.70 | |
| GoogLeNet BN After Ortho |0.584 | 1.77  | |


![CaffeNet128 test accuracy](/logs/bn_init/img/0caffe.png)


![CaffeNet128 test loss](/logs/bn_init/img/2caffe.png)


![CaffeNet128 train loss](/logs/bn_init/img/6caffe.png)


![GoogleNet128 test accuracy](/logs/bn_init/img/0google.png)


![GoogleNet128 test loss](/logs/bn_init/img/2google.png)


![GoogleNet128 train loss](/logs/bn_init/img/6google.png)
    
[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/bn_init), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/bn_init)


### BatchNorm evaluation ReLU

![CaffeNet128 test accuracy](/logs/img/0.png)


![CaffeNet128 test loss](/logs/img/2.png)


![CaffeNet128 train loss](/logs/img/6.png)


### Different activations plus BN
As one can see, BN makes difference between ReLU, ELU and PReLU negligable. 
It may confirm that main source of VLReLU and ELU advantages is that their output is closer to mean=0, var=1, than standard ReLU.

![CaffeNet128 test accuracy](/logs/img/bn_act0.png)


![CaffeNet128 test loss](/logs/img/bn_act2.png)


![CaffeNet128 train loss](/logs/img/bn_act6.png)

### Batch Normalization and Dropout
BN+Dropout = 0.5 is too much regularization. Dropout=0.2 is just enough :)
![CaffeNet128 test accuracy](/logs/img/bn_dropout0.png)


![CaffeNet128 test loss](/logs/img/bn_dropout2.png)


![CaffeNet128 train loss](/logs/img/bn_dropout6.png)


### Do we need EltwiseAffine layer?

![CaffeNet128 test accuracy](/logs/img/0_EA.png)

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Anything vs. seconds" will give weird results. 
