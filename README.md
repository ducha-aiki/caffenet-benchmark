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

![CaffeNet128 test accuracy](/logs/img/beforeReLU_ea.png)

TBD: Explore usefullness of BatchNorm+[EltwiseAffine](https://github.com/BVLC/caffe/pull/2996) combination

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Anything vs. seconds" will give weird results. 

