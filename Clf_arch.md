This is quick evaluation of different classifier (fc6-fc8) designs performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### CLF architecture

pool5pad *Nets mistakenly were trained with ELU non-linearity instead of default ReLU

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default ELU|0.488| 2.28 | fc6 = conv 3x3x2048 -> fc7 2048 -> 1000 fc8 |
| pool5pad_fc6ave |0.481| 2.32 | pool5 zero pad -> fc6 = conv 3x3x2048 -> AvePool -> as usual |
| pool5pad_fc6ave_fc7as1x1 | **0.511** | 2.21 | pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> fc8 as 1x1 conv -> ave_pool.  |
| pool5pad_fc6ave_fc7as1x1_avemax_fc8 | 0.509 | **2.19** | pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> fc8 as 1x1 conv -> ave_pool + max_pool. |
| Default ReLU|0.470| 2.36 | fc6 = conv 3x3x2048 -> fc7 2048 -> 1000 fc8 |
| Default ReLU|0.497| 2.24 | fc6 = conv 3x3x4096 -> fc7 4096 -> 1000 fc8 == original caffenet|
| fc6=512C3_1024C3_1536C1 |0.482| 2.52 | pool5 zero pad -> fc6 = conv 3x3x512 -> fc7=conv 3x3x1024 -> 1x1x1536 ->  fc8 as 1x1 conv -> ave_pool.  |
| fc6=512C3_1024C3_1536C1_drop |0.491| 2.29 | pool5 zero pad -> fc6 = conv 3x3x512 -> fc7=conv 3x3x1024 -> drop 0.3 -> 1x1x1536 -> drop 0.5-> fc8 as 1x1 conv -> ave_pool.  |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/clf_arch), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/clf_arch)



![CaffeNet128 test accuracy](/logs/clf_arch/img/0.png)


![CaffeNet128 test loss](/logs/clf_arch/img/2.png)


![CaffeNet128 train loss](/logs/clf_arch/img/6.png)

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

