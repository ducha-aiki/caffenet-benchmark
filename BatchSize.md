This is quick evaluation of batch size influence on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)


### Batch size, ReLU
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| BS=1024, 4xlr | 0.465| 2.38 | lr=0.04, 80K iters |
| BS=1024 | 0.419| 2.65 | lr=0.01, 80K iters |
| BS=512, 2xlr |0.469| 2.37 | lr=0.02, 160K iters |
| BS=512 |0.455| 2.46 | lr=0.01, 160K iters |
| BS=256, **default** |0.471| 2.36 |  lr=0.01, 320K iters |
| BS=128 |0.472| 2.35 | lr=0.01, 640K iters |
| BS=128, 1/2 lr | 0.470 | 2.36| lr=0.005, 640K iters |
| BS=64| 0.471| **2.34** |  lr=0.01, 1280K iters |
| BS=64, 1/4 lr| **0.475** | **2.34** |  lr=0.0025, 1280K iters |
| BS=32 |0.463| 2.40 | lr=0.01, 2560K iter |
| BS=32, 1/8 lr| 0.470 | 2.37|  lr=0.00125,  2560K iter|
| BS=1, 1/256 lr| 0.474 | 2.35|  lr=3.9063e-05,  81920K iter. Online training|

Graph below shows test accuracy as function of #epoch, so scale is batch_size independent.

![CaffeNet128 test accuracy](/logs/batch_size/img/0.png)

So general recommendation: too big batch_sizes leads to a bit inferior results, but in general batch_size should be selected based computation speed. If learning rate is adjusted, than no practial difference between different batch sizes.


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


