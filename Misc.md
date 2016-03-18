This is quick evaluation of everything, which not fit in any other category on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### Other

ReLU non-linearity, fc6 and fc7 layer only

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
|  Default |0.471| 2.36 | 290K iters stopped|
|  NoBias |0.445| 2.50 | Biases initialized with zeros, lr_rate = 0|

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/other), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/other)



![CaffeNet128 test accuracy](/logs/other/img/0.png)


![CaffeNet128 test loss](/logs/other/img/2.png)


![CaffeNet128 train loss](/logs/other/img/6.png)



P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 


