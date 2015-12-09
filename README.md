This is quick evaluation of BatchNorm layer (https://github.com/BVLC/caffe/pull/3229) performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)

Because LRN layers add nothing to accuracy, they were removed for speed reasons in further experiments.

### BatchNorm evaluation

![CaffeNet128 test accuracy](/logs/img/0.png)


![CaffeNet128 test loss](/logs/img/2.png)


![CaffeNet128 train loss](/logs/img/6.png)


### Do we need EltwiseAffine layer?

![CaffeNet128 test accuracy](/logs/img/beforeReLU_ea.png)

TBD: Explore usefullness of BatchNorm+[EltwiseAffine](https://github.com/BVLC/caffe/pull/2996) combination

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Anything vs. seconds" will give weird results. 

