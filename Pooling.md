This is quick evaluation of different pooling functions performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### Pooling

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| MaxPool |0.470| 2.36 | |
| [Stochastic](http://arxiv.org/abs/1301.3557) |0.438| 2.54| Underfitting, may be try without Dropout|
| AvgPool |0.435| 2.56 | |
| Max+AvgPool |**0.483**| **2.29** | Element-wise sum|


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/pooling), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/pooling)



![CaffeNet128 test accuracy](/logs/pooling/img/0.png)


![CaffeNet128 test loss](/logs/pooling/img/2.png)


![CaffeNet128 train loss](/logs/pooling/img/6.png)

Previous results on small datasets like CIFAR (see [LSUV-init, Table3](http://arxiv.org/abs/1511.06422)) looks a bit contradictory to ImageNet ones so far.

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

