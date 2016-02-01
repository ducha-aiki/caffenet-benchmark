This is quick evaluation of different classifier (fc6-fc8) designs performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### CLF architecture

Nets mistakenly were trained with ELU non-linearity instead of default ReLU

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default |0.488| 2.28 | fc6 = conv 3x3x2048 -> fc7 2048 -> 1000 fc8 |
| pool5pad_fc6ave |0.481| 2.32 | pool5 zero pad -> fc6 = conv 3x3x2048 -> AvePool -> as usual |
| pool5pad_fc6ave_fc7as1x1 | **0.511** | 2.21 | pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> fc8 as 1x1 conv -> ave_pool.  |
| pool5pad_fc6ave_fc7as1x1_avemax_fc8 | 0.509 | **2.19** | pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> fc8 as 1x1 conv -> ave_pool + max_pool. |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/clf_arch), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/clf_arch)



![CaffeNet128 test accuracy](/logs/clf_arch/img/0.png)


![CaffeNet128 test loss](/logs/clf_arch/img/2.png)


![CaffeNet128 train loss](/logs/clf_arch/img/6.png)

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

