Welcome to evaluation of CNN design choises performance on ImageNet-2012. Here you can find prototxt`s of tested nets and full train logs.

The basic architecture is similar to CaffeNet, but has several differences:

1. Images are resized to small side = 128 for speed reasons. Therefore pool5 spatial size is 3x3 instead of 6x6.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. Because LRN layers add nothing to accuracy (validated [here](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)), they were removed for speed reasons in most experiments.

On-going evaluations with graphs:
- [activations](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Activations.md)
- [pooling](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Pooling.md)
- [architectures] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Architectures.md)
- [augmentation] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Augmentation.md)
- [batchnorm] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)

### Activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [ReLU](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf) |0.470| 2.36 | With LRN layers|
| ReLU |0.470| 2.36 | No LRN, as in rest |
| [VLReLU](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf) |0.469| 2.40|y=max(x,x/3)|
| [RReLU](http://arxiv.org/abs/1505.00853) |0.478| 2.32| |
| [Maxout](http://arxiv.org/abs/1302.4389) |0.482| 2.30| sqrt(2) narrower layers, 2 pieces|
| [PReLU](http://arxiv.org/abs/1502.01852) |0.485| 2.29 | |
| [ELU](http://arxiv.org/abs/1511.07289) |**0.488**| **2.28**| |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/activations), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/activations)

### Pooling

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| MaxPool |0.470| 2.36 | |
| [Stochastic](http://arxiv.org/abs/1301.3557) |0.438| 2.54| Underfitting, may be try without Dropout|
| AvgPool |0.435| 2.56 | |



[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/pooling), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/pooling)

### Architectures

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [CaffeNet256](http://arxiv.org/abs/1408.5093) |0.565| 1.87 | Reference BVLC model, LSUV init|
| [CaffeNet](http://arxiv.org/abs/1408.5093) |0.470| 2.36 | Pool5 = 3x3|
| [CaffeNetSPP](http://arxiv.org/abs/1406.4729) |0.483| 2.30 | SPP= 3x3 + 2x2 + 1x1 |
| [HeNet](http://arxiv.org/abs/1412.1710) |**0.561**| **1.88**|No SPP, Pool5 = 3x3, VLReLU |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.407| 2.80| With BN, huge overfitting, no dropout. First attempt to train, probably smth went wrong |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.518| 2.16| Without BN, less overfitting (still > than HeNet), no dropout. Looks like needs bigger initial LR, than caffenet. 3rd attempt is on-going  |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.525| 2.12| Without BN, less overfitting (still > than HeNet), no dropout. LR=0.05 until 25K iters.  |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/architectures), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/architectures)

### Train augmentation

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default |0.470| 2.36 | Random flip, random crop 128x128 from 144xN, N > 144|
| Drop 0.1 |0.306| 3.56 | + Input dropout 10%. not finished, 186K iters result |
| Multiscale |0.462| 2.40 | Random flip, random crop 128x128 from ( 144xN, - 50%, 188xN - 20%, 256xN - 20%, 130xN - 10%)


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/augmentation), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/augmentation)

## Batch normalization
[BN-paper](http://arxiv.org/abs/1502.03167), [caffe-PR](https://github.com/BVLC/caffe/pull/3229)
Note, that results are obtained without mentioned in paper y=kx+b additional layer.

### BN -- before or after ReLU?
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Before |0.474| 2.35 | As in paper|
| After |**0.499**| **2.21** | |

So in all next experiments, BN is put after non-linearity

### BN and activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| ReLU |0.499| 2.21 | |
| RReLU |0.500| 2.20 | |
| PReLU |**0.503**| **2.19** | |
| ELU |0.498| 2.23 | |
| Maxout |0.487| 2.28| |

### BN and dropout

ReLU non-linearity, fc6 and fc7 layer only

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
|  Dropout = 0.5 |0.499| 2.21 |  |
|  Dropout = 0.2 |**0.527**| **2.09** |  |
|  Dropout = 0 |0.513| 2.19 | |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs)

The PRs with test are welcomed

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Anything vs. seconds" will give weird results. 

