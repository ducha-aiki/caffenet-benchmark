This is quick evaluation of different colorspace preprocessing performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.




![CaffeNet128 test accuracy](/logs/colorspace/img/0.png)


![CaffeNet128 test loss](/logs/colorspace/img/2.png)


![CaffeNet128 train loss](/logs/colorspace/img/6.png)

