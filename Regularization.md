This is quick evaluation of different regularization strategies performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### Regularization

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| default |0.470| 2.36 | weight_decay=0.0005, L2, fc-dropout=0.5 |
| wd=0.0001 |0.450| 2.48 | weight_decay=0.0001, L2, fc-dropout=0.5 |
| wd=0.00001 |0.450| 2.48 | weight_decay=0.00001, L2, fc-dropout=0.5 |
| wd=0.00001_L1 |0.453| 2.45 | weight_decay=0.00001, L1, fc-dropout=0.5 |
| drop=0.3 |0.471* | 2.38* | Not finished yet, 170K iters. weight_decay=0.0005, L2, fc-dropout=0.3 |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/regularization), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/regularization)



![CaffeNet128 test accuracy](/logs/regularization/img/0.png)


![CaffeNet128 test loss](/logs/regularization/img/2.png)


![CaffeNet128 train loss](/logs/regularization/img/6.png)

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

