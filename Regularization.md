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
| drop=0.3 | **0.497** | **2.25** | weight_decay=0.0005, L2, fc-dropout=0.3 |
| drop=0.2 | 0.494 | 2.28 | weight_decay=0.0005, L2, fc-dropout=0.2 |
| drop=0.1 | 0.473 | 2.45 | weight_decay=0.0005, L2, fc-dropout=0.1. Same acc, as in 0.5, but bigger logloss |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/regularization), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/regularization)



### Dropout and width 

Hypothesis about "same effective neurons = same performance" looks unvalidated

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| fc6,fc7=2048, dropout=0.5 |0.470| 2.36 | default |
| fc6,fc7=2048, dropout=0.3 | **0.497** | 2.25 | best for fc6,fc7=2048. 0.7*2048=1433 neurons work each time |
| fc6,fc7=4096, dropout=0.65 |0.465| 2.38 | 0.65*4096=1433 neurons work each time |
| fc6,fc7=6144, dropout=0.77 |0.447| 2.48 | 0.77*6144=1433 neurons work each time |
| fc6,fc7=4096, dropout=0.5 | **0.497** | **2.24** | |
| fc6,fc7=1433, dropout=0 | 0.456 | 2.52 | |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/regularization/dropout_same_effective), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/regularization/dropout_same_effective)




![CaffeNet128 test accuracy](/logs/regularization/img/0.png)


![CaffeNet128 test loss](/logs/regularization/img/2.png)


![CaffeNet128 train loss](/logs/regularization/img/6.png)


![CaffeNet128 test accuracy](/logs/regularization/dropout_same_effective/img/0.png)


![CaffeNet128 test loss](/logs/regularization/dropout_same_effective/img/2.png)


![CaffeNet128 train loss](/logs/regularization/dropout_same_effective/img/6.png)


P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

