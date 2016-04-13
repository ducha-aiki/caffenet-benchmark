This is quick evaluation of batch size influence on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### Batch size, ReLU
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| BS=1024 | 0.419| 2.65 | lr=0.01, 80K iters |
| BS=512 |0.455| 2.46 | lr=0.01, 160K iters |
| BS=256, **default** |0.471| 2.36 |  lr=0.01, 320K iters |
| BS=128 |0.472| 2.35 | lr=0.01, 640K iters |
| BS=128, 1/2 lr | 0.470 | 2.36| lr=0.005, 640K iters |
| BS=64|-| -|  lr=0.01, 1280K iters, in progress |
| BS=64, 1/4 lr| **0.475** | **2.34** |  lr=0.0025, 1280K iters |
| BS=32 |0.463| 2.40 | lr=0.01, 2560K iter |
| BS=32, 1/8 lr| 0.470 | 2.37|  lr=0.00125,  2560K iter|


![CaffeNet128 test accuracy](/logs/batch_size/img/0.png)

![CaffeNet128 test loss](/logs/batch_size/img/2.png)

![CaffeNet128 train loss](/logs/batch_size/img/6.png)



[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/batch_size), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/batch_size)

### Batch size, TanH.

n/c - not converged (in starting iterations)

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| BS=256 |0.401| 2.78 | **TanH**, lr=0.01, 320K iters |
| BS=128 |0.407| 2.77 | lr=0.01, 640K iters |
| BS=128, 2xlr | n/c | n/c | lr=0.02 |
| BS=128, 1/2 lr | 0.402 | 2.77| lr=0.005 |
| BS=64 | 0.384| 2.89 | lr=0.01, 1280K iters |
| BS=64, 4xlr | n/c| n/c |  lr=0.04 |
| BS=64, 1/4 lr| 0.401 | 2.78|  lr=0.0025 |
| BS=32 |0.373| 2.95 | lr=0.01, 2560K iter
| BS=32, 8xlr | n/c | n/c | lr=0.08 |
| BS=32, 1/8 lr| - | -|  lr=0.00125, in progress |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/batch_size/tanh), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/batch_size/tanh)



P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 


