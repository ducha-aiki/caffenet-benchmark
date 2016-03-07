This is quick evaluation of different pooling functions performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### Pooling type

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| MaxPool |0.470| 2.36 | 290K iters stopped|
| [Stochastic](http://arxiv.org/abs/1301.3557) |0.438| 2.54| Underfitting, may be try without Dropout|
| AvgPool |0.435| 2.56 | |
| Max+AvgPool | **0.483** | **2.29** | Element-wise sum|
| NoPool |0.472| 2.35 | Strided conv2,conv3,conv4 |


### Pooling window/stride

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| MaxPool 3x3/2 |0.470| 2.36 | default alexnet|
| MaxPool 2x2/2 | 0.484 | 2.29 |Leads to larger feature map, Pool5=4x4 instead of  3x3 |
| MaxPool 3x3/2 pad1 | **0.488** | **2.25** |Leads to even larger feature map, Pool5=5x5 instead of  3x3 |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/pool), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/pooling)

![CaffeNet128 test accuracy](/logs/pooling/img/0.png)


![CaffeNet128 test loss](/logs/pooling/img/2.png)


![CaffeNet128 train loss](/logs/pooling/img/6.png)

### General pooling testing
Authors of [Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree](http://arxiv.org/abs/1509.08985) kindly provided reference implementation for test. 
Unfortunately, under patent: UCSD Docket No. SD2016-053, "Generalizing Pooling Functions in Convolutional Neural Network", filed on Sept 23, 201

The performance is good, but seems dependent on other design choises (i.e. beat MaxPool in one setup and lose in another) and also on initialization.

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| MaxPool128-2048 |0.470| 2.36 | My reference caffenet128 |
| GatedAveMaxPool128-2048 |0.471| 2.36 | |
| GeneralPool128-2048 |0.464*| 2.46* | Unfinished, 227K iters |
| MaxPool128-4096 |0.497| 2.24 | fc6,fc7 = 4096 |
| GeneralPool128-4096 |0.494| 2.25 |fc6,fc7 = 4096  |
| MaxPool227-4096 |0.565| 1.87 | My reference caffenet227 |
| GeneralPool227-4096 |0.570| 1.86 |  |
| Authors GeneralPool227-4096 |0.585| 1.78 | Different lr_policy: each step is longer |

![CaffeNet128 test accuracy](/logs/pooling/gen_pool/img/0.png)

![CaffeNet128 test loss](/logs/pooling/gen_pool/img/2.png)

![CaffeNet128 train loss](/logs/pooling/gen_pool/img/6.png)

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/pool/gen_pool), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/pooling)



Previous results on small datasets like CIFAR (see [LSUV-init, Table3](http://arxiv.org/abs/1511.06422)) looks a bit contradictory to ImageNet ones so far.

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

