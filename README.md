Welcome to evaluation of CNN design choises performance on ImageNet-2012. Here you can find prototxt`s of tested nets and full train logs.

The basic architecture is similar to CaffeNet, but has several differences:

1. Images are resized to small side = 128 for speed reasons. Therefore pool5 spatial size is 3x3 instead of 6x6.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. Because LRN layers add nothing to accuracy (validated [here](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)), they were removed for speed reasons in most experiments.

Taking into account http://beta.openreview.net/forum?id=OM0jKROjrFp57ZJjtNkv, results can vary from run to run (data order is the same, but random seeds are different). However, I haven`t experienced results difference for several CaffeNet-ReLU training runs.

On-going evaluations with graphs:
- [activations](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Activations.md)
- [pooling](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Pooling.md)
- [solvers](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Solvers.md)
- [lr_policy](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Lr_policy.md)
- [architectures] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Architectures.md)
- [First layer parameters] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/First_layer_input.md)
- [classfier architectures] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Clf_arch.md)
- [augmentation] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Augmentation.md)
- [batchnorm] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)
- [colorspace] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Colorspace.md)
- [regularization] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Regularization.md)
- [resnets, not yet successfull] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/ResNets.md)
- [batch size] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/BatchSize.md)
- [dataset size] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Dataset_size.md)
- [Network width] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Complexity.md)
- [other mix] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Misc.md)


### Activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [ReLU](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf) |0.470| 2.36 | With LRN layers|
| ReLU |0.471| 2.36 | No LRN, as in rest |
| TanH |0.401| 2.78 |  |
| [1.73TanH(2x/3)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) | 0.423 |  2.66 | As recommended in Efficient BackProp, LeCun98  |
| [VLReLU](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf) |0.469| 2.40|y=max(x,x/3)|
| [RReLU](http://arxiv.org/abs/1505.00853) |0.478| 2.32| |
| [Maxout](http://arxiv.org/abs/1302.4389) |0.482| 2.30| sqrt(2) narrower layers, 2 pieces. Same complexity, as for ReLU|
| [Maxout](http://arxiv.org/abs/1302.4389) | *0.517* | *2.12* | same width layers, 2 pieces|
| [PReLU](http://arxiv.org/abs/1502.01852) |0.485| 2.29 | |
| [ELU](http://arxiv.org/abs/1511.07289) |0.488| 2.28| |
| Shifted [Softplus](http://machinelearning.wustl.edu/mlpapers/papers/AISTATS2011_GlorotBB11) |0.486| 2.29| Shifted BNLL aka softplus, y = log(1 + exp(x)) - log(2). Same as ELU, as expected |
| No |0.389 | 2.93 | No non-linearity |
| [APL](http://arxiv.org/abs/1412.6830)2 |0.471 | 2.38 | 2 linear pieces. Unlike other activations, [current author`s implementation](https://github.com/forestagostinelli/Learned-Activation-Functions-Source/issues/4) leads to different parameters for each x,y position of neuron |
| [APL](http://arxiv.org/abs/1412.6830)5 |0.465 | 2.39 | 5 linear pieces. Unlike other activations, [current author`s implementation](https://github.com/forestagostinelli/Learned-Activation-Functions-Source/issues/4) leads to different parameters for each x,y position of neuron |
| ConvReLU,FCMaxout2 | 0.490 | 2.26 | ReLU in convolution, Maxout (sqrt(2) narrower) 2 pieces in FC. Inspired by kaggle and [INVESTIGATION OF MAXOUT NETWORKS FOR SPEECH RECOGNITION](http://www.cstr.ed.ac.uk/downloads/publications/2014/Swietojanski_ICASSP14.pdf)*  |
| ConvELU,FCMaxout2 | **0.499** | **2.22** | ELU in convolution, Maxout (sqrt(2) narrower) 2 pieces in FC.  |

*
>The above analyses show that the bottom layers seem to waste
>a large portion of the additional parametrisation (figure 2 (a,e)) thus
>could be replaced, for example, by smaller ReLU layers. Similarly,
>maxout units in higher layers seem to use piecewise-linear components
>in a more active way suggesting the use of larger pools._


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/activations), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/activations)

### Pooling type

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| MaxPool |0.471| 2.36 | 290K iters stopped|
| [Stochastic](http://arxiv.org/abs/1301.3557) |0.438| 2.54| Underfitting, may be try without Dropout|
| Stochastic, no dropout |0.429| 2.96| Stoch pool does not prevent overfitting without dropout :(. Good start,bad finish|
| AvgPool |0.435| 2.56 | |
| Max+AvgPool | **0.483** | **2.29** | Element-wise sum|
| NoPool |0.472| 2.35 | Strided conv2,conv3,conv4 |
| [General](http://arxiv.org/abs/1509.08985) | - | - | Depends on arch, [click for details](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Pooling.md)|

### Pooling window/stride

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| MaxPool 3x3/2 |0.471| 2.36 | default alexnet|
| MaxPool 2x2/2 | 0.484 | 2.29 |Leads to larger feature map, Pool5=4x4 instead of  3x3 |
| MaxPool 3x3/2 pad1 | **0.488** | **2.25** |Leads to even larger feature map, Pool5=5x5 instead of  3x3 |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/pooling), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/pooling)

### CLF architecture

pool5pad *Nets mistakenly were trained with ELU non-linearity instead of default ReLU

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default ELU|0.488| 2.28 | fc6 = conv 3x3x2048 -> fc7 2048 -> 1000 fc8 |
| pool5pad_fc6ave |0.481| 2.32 | pool5 zero pad -> fc6 = conv 3x3x2048 -> AvePool -> as usual |
| pool5pad_fc6ave_fc7as1x1fc8ave | 0.508 | 2.22 |  pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> ave_pool -> fc8 as 1x1 conv.  |
| pool5pad_fc6ave_fc7as1x1avefc8 | **0.511** | 2.21 |pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> fc8 as 1x1 conv -> ave_pool.  |
| pool5pad_fc6ave_fc7as1x1_avemax_fc8 | 0.509 | **2.19** | pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> fc8 as 1x1 conv -> ave_pool + max_pool. |
| Default ReLU|0.471| 2.36 | fc6 = conv 3x3x2048 -> fc7 2048 -> 1000 fc8 |
| fc6-7 4096|0.497| 2.24 | fc6 = conv 3x3x4096 -> fc7 4096 -> 1000 fc8 == original caffenet|
| fc6=512C3_1024C3_1536C1 |0.482| 2.52 | pool5 zero pad -> fc6 = conv 3x3x512 -> fc7=conv 3x3x1024 -> 1x1x1536 ->  fc8 as 1x1 conv -> ave_pool.  |
| fc6=512C3_1024C3_1536C1_drop |0.491| 2.29 | pool5 zero pad -> fc6 = conv 3x3x512 -> fc7=conv 3x3x1024 -> drop 0.3 -> 1x1x1536 -> drop 0.5-> fc8 as 1x1 conv -> ave_pool.  |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/clf_arch), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/clf_arch)

### Conv1 parameters

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| ------:|:-----------|
| Default, 128_K11_S4 |  **0.471** | 2.36 | Input size =128x128px, conv1 = 11x11x96, stride = 4 |
| 224_K11_S8 |0.453| 2.45 | Input size =256x256px, conv1 = 11x11x96, stride = 8. Not finished yet |
| 160_K11_S5 | 0.470| **2.35** | Input size =160x160px, conv1 = 11x11x96, stride = 5 |
| 96_K7_S3 | 0.459 | 2.43 | Input size =96x96px, conv1 = 7x7x96, stride = 3 |
| 64_K5_S2 | 0.445 | 2.50 | Input size =64x64px, conv1 = 5x5x96, stride = 2 |
| 32_K3_S1| 0.390 | 2.84 | Input size =32x32px, conv1 = 3x3x96, stride = 1 |
| 4x slower, 227_K11_S4 | 0.565 | 1.87 | Input size = 227x227px, conv1 = 11x11x96, stride = 4,  Not finished yet |

[prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/conv1-input), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/conv1-input)

## Squeezing representation

For example, for using activations in image retrieval. 

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| pool5pad_fc6ave_fc7as1x1fc8ave | 0.508 | 2.22 |  Baseline. pool5 zero pad -> fc6 = conv 3x3x2048 ->  fc7 as 1x1 conv -> ave_pool -> fc8 as 1x1 conv.  |
| pool5pad_fc6ave_fc7as1x1=512_fc8ave | 0.489 | 2.30 |  fc7 as 1x1 conv  = 512   |
| pool5pad_fc6ave_fc7as1x1_bottleneck=512_fc8ave | 0.490 | 2.28 | fc7 as 1x1 conv = 2048 then fc7a = 512  |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/clf_arch/bottleneck), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/clf_arch/bottleneck)

### Solvers

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| SGD with momentum |0.471| 2.36 | |
| Nesterov |**0.473**|**2.34**|  |
| RMSProp |0.327| 3.20 | rms_decay=0.9, delta=1.0  |
| RMSProp | 0.453| 2.45 | rms_decay=0.9, delta=1.0, base_lr: 0.045, stepsize=10K. gamma=0.94 (from [here](https://github.com/smichalowski/google_inception_v3_for_caffe)) |
| RMSProp | 0.451| 2.43| rms_decay=0.9, delta=1.0, base_lr: 0.1, stepsize=10K. gamma=0.94 |
| RMSProp | 0.472| 2.36| rms_decay=0.9, delta=1.0, base_lr: 0.1, stepsize=5K. gamma=0.94 |
| RMSProp |0.486| 2.28 | rms_decay=0.9, delta=1.0, lr=0.1, linear lr_policy  |
| SGD with momentum, linear |0.493| 2.24 | linear lr_policy  |

Not converge at all:

ADAM: 
lr=0.001 m=0.9 m2=0.999 delta=1e-8
lr=0.001 m=0.95 m2=0.999 delta=1e-8
lr=0.001 m=0.95 m2=0.999 delta=1e-7
lr=0.01 m=0.9 m2=0.999 delta=1e-8
lr=0.01 m=0.9 m2=0.999 delta=1e-7
lr=0.01 m=0.9 m2=0.999 delta=1e-9
lr=0.01 m=0.9 m2=0.99 delta=1e-8
lr=0.01 m=0.9 m2=0.999 delta=1e-8
lr=0.01 m=0.95 m2=0.999 delta=1e-9

AdaDelta:
delta: 1e-5

RMSProp, 
lr=0.01, rms_decay=0.5
lr=0.01, rms_decay=0.9
lr=0.01, rms_decay=0.95
lr=0.01, rms_decay=0.98
lr=0.001, rms_decay=0.9
lr=0.001, rms_decay=0.98

Converge, but much worse that SGD:
Adagrad, lr=0.01, lr=0.02
AdaDelta: delta: 1e-6, delta: 1e-7, delta: 1e-8
RMSProp, lr=0.01, rms_decay=0.99


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/solvers), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/solvers)

### LR-policy

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Step 100K |0.471| 2.36 | Default caffenet solver, max_iter=320K |
| Poly lr, p=0.5, sqrt |0.483| 2.29 | bvlc_quick_googlenet_solver, All the way worse than "step", leading at finish |
| Poly lr, p=2.0, sqr |0.483| 2.299 | |
| Poly lr, p=1.0, linear |**0.493**|***2.24*** | |
| Poly lr, p=1.0, linear |0.466|2.39 | max_iter=160K|
| Exp, 0.035 |0.441|2.53 | max_iter=160K, stepsize=2K, gamma=0.915, same as in base_dereyly|

#### LR-policy-BatchNorm-Dropout = 0.2

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Step 100K |**0.527**| **2.09** | Default caffenet solver, max_iter=320K |
| Poly lr, p=1.0, linear |0.496|2.24 | max_iter=105K,|
| Poly lr, p=1.0, start_lr=0.02 |0.505| 2.21 | max_iter=105K|
| Exp, 0.035 |0.506| 2.19 | max_iter=160K, stepsize=2K, gamma=0.915, same as in base_dereyly|


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/lr_policy), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/lr_policy)


### Regularization

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| default |0.471| 2.36 | weight_decay=0.0005, L2, fc-dropout=0.5 |
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
| fc6,fc7=2048, dropout=0.5 |0.471| 2.36 | default |
| fc6,fc7=2048, dropout=0.3 | **0.497** | 2.25 | best for fc6,fc7=2048. (1-0.3)*2048=1433 neurons work each time |
| fc6,fc7=4096, dropout=0.65 |0.465| 2.38 | (1-0.65)*4096=1433 neurons work each time |
| fc6,fc7=6144, dropout=0.77 |0.447| 2.48 | (1-0.77)*6144=1433 neurons work each time |
| fc6,fc7=4096, dropout=0.5 | **0.497** | **2.24** | |
| fc6,fc7=1433, dropout=0 | 0.456 | 2.52 | |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/regularization/dropout_same_effective), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/regularization/dropout_same_effective)


### Architectures

CaffeNet only

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [CaffeNet256](http://arxiv.org/abs/1408.5093) | **0.565** | **1.87** | Reference BVLC model, LSUV init|
| [CaffeNet128](http://arxiv.org/abs/1408.5093) |0.470| 2.36 | Pool5 = 3x3|
| [CaffeNet128_4096](http://arxiv.org/abs/1408.5093) |0.497| 2.24 | Pool5 = 3x3, fc6-fc7=4096|
| CaffeNet128All | **0.530** | **2.05** | All improvements without caffenet arch change: ELU + SPP + color_trans3-10-3 + Nesterov+ (AVE+MAX) Pool + linear lr_policy|
|  | + 0.06 |  | Gain over vanilla caffenet128. "Sum of gains" = 0.018 + 0.013 + 0.015 + 0.003 + 0.013 + 0.023 = 0.085 |
| [SqueezeNet128](https://github.com/DeepScale/SqueezeNet) | **0.530** | 2.08 | Reference solver, but linear lr_policy and batch_size=256 (320K iters). WITHOUT tricks like ELU, SPP, AVE+MAX, etc.|
| SqueezeNet128+ELU | **0.555** | **1.95** | Reference solver, but linear lr_policy and batch_size=256 (320K iters).ELU|
| CaffeNet256All | 0.613 | 1.64 | All improvements without caffenet arch change: ELU + SPP + color_trans3-10-3 + Nesterov+ (AVE+MAX) Pool + linear lr_policy|
| CaffeNet128, no pad |0.411 | 2.70 | No padding, but conv1 stride=2 instead of 4 to keep size of pool5 the same|
| CaffeNet128, dropout in conv | 0.426| 2.60 | Dropout before pool2=0.1, after conv3 = 0.1, after conv4 = 0.2 |
| [CaffeNet128SPP](http://arxiv.org/abs/1406.4729) |0.483| 2.30 | SPP= 3x3 + 2x2 + 1x1 |
| [DarkNet128BN](http://pjreddie.com/darknet/imagenet/#reference) |0.502| 2.25 | 16C3->MP2->32C3->MP2->64C3->MP2->128C3->MP2->256C3->MP2->512C3->MP2->1024C3->1000CLF.BN|
| | | | + PreLU + base_lr=0.035, exp lr_policy, 160K iters |
| [NiN128](http://arxiv.org/abs/1312.4400) |0.519 |  2.15 | Step lr_policy. Be carefull to not use dropout on maxpool in-place|


Others

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [DarkNetBN](http://pjreddie.com/darknet/imagenet/#reference) |0.502| 2.25 | 16C3->MP2->32C3->MP2->64C3->MP2->128C3->MP2->256C3->MP2->512C3->MP2->1024C3->1000CLF.BN|
| [HeNet2x2](http://arxiv.org/abs/1412.1710) |0.561 | 1.88|No SPP, Pool5 = 3x3, VLReLU, J' from paper |
| [HeNet3x1](http://arxiv.org/abs/1412.1710) |0.560 | 1.88|No SPP, Pool5 = 3x3, VLReLU, J' from paper, 2x2->3x1 |
| [GoogLeNet128](http://arxiv.org/abs/1409.4842) | **0.619** | **1.61** |linear lr_policy, batch_size=256. obviously slower than caffenet |
| googlenet_loss2_clf| 0.571 | 1.80 | from net above, aux classifier after inception_4d |
| googlenet_loss1_clf| 0.520 | 2.06 | from net above, aux classifier after inception_4a |
| [fitnet1_elu](http://arxiv.org/abs/1412.6550) | 0.333 | 3.21 | |


ResNet attempts are moved to [ResNets.md](ResNets.md)

### ResNets, good attempts


| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [ResNet-50ELU-2xThinner](http://arxiv.org/abs/1512.03385) | 0.616| 1.63| Without BN, ELU, dropout=0.2 before classifier. 2x thinner, than in paper. Quite fast. No large overfitting (unlike upper table) |
| [GoogLeNet-128](http://arxiv.org/abs/1409.4842) | **0.619** | **1.61** | For reference. linear lr_policy, batch_size=256. |


### Train augmentation

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default |**0.471**| **2.36** | Random flip, random crop 128x128 from 144xN, N > 144|
| Drop 0.1 |0.306| 3.56 | + Input dropout 10%. not finished, 186K iters result |
| Multiscale |0.462| 2.40 | Random flip, random crop 128x128 from ( 144xN, - 50%, 188xN - 20%, 256xN - 20%, 130xN - 10%)
| 5 deg rot |0.448| 2.47 | Random rotation to [0..5] degrees. |
    
[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/augmentation), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/augmentation)

### Colorspace

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| RGB |0.471| 2.36 | default, no changes. Input = 0.04 * (Img - [104, 117,124]) |
| RGB_by_BN |0.469| 2.38 | Input = BatchNorm(Img)|
| CLAHE| 0.467 | 2.38 | RGB -> LAB -> CLAHE(L)->RGB->BatchNorm(RGB) |
| HISTEQ| 0.448 | 2.48 | RGB -> HiestEq |
| YCrCb | 0.458| 2.42 | RGB->YCrCb->BatchNorm(YCrCb)  |
| HSV |0.451| 2.46 |RGB->HSV->BatchNorm(HSV)  |
| Lab |-| - | Doesn`t leave 6.90 loss after 1.5K iters |
| RGB->10->3 TanH | 0.463| 2.40 | RGB -> conv1x1x10 tanh  ->  conv1x1x3 tanh |
| RGB->10->3 VlReLU | **0.485** | **2.28** | RGB -> conv1x1x10 vlrelu  ->  conv1x1x3 vlrelu|
| RGB->16->3 VlReLU | 0.483 | 2.30 | RGB -> conv1x1x16 vlrelu  ->  conv1x1x3 vlrelu|
| RGB->3->3 VlReLU | 0.480 | 2.32 | RGB -> conv1x1x3 vlrelu  ->  conv1x1x3 vlrelu|
| RGB->10->3 VlReLU->sum(RGB) | 0.482 | 2.30 | RGB -> conv1x1x10 vlrelu  ->  conv1x1x3 -> sum(RGB) ->vlrelu|
| RGB and log(RGB)->10->3 VlReLU | 0.482 | 0.482 | RGB and log (RGB) -> conv1x1x10 vlrelu  ->  conv1x1x3 vlrelu|
| NN-Scale |0.467| 2.38 | Nearest neightbor instead of linear interpolation for rescale. Faster, but worse :(|
| concat_rgb_each_pool |0.441| 2.51 | Concat avepoolRGB with each pool |
| OpenCV RGB2Gray |0.413| 2.70 |RGB->Grayscale Gray = 0.299 R + 0.587 G + 0.114 B |
| Learned RGB2Gray |0.419| 2.66 |RGB->conv1x1x1  |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/colorspace), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/colorspace)


## Batch normalization
[BN-paper](http://arxiv.org/abs/1502.03167), [caffe-PR](https://github.com/BVLC/caffe/pull/3229)
Note, that results are obtained without mentioned in paper y=kx+b additional layer.

### BN -- before or after ReLU?
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Before |0.474| 2.35 | As in paper|
| Before + scale&bias layer |0.478| 2.33 | As in paper|
| After |**0.499**| **2.21** | |
| After + scale&bias layer |0.493| 2.24 | |

So in all next experiments, BN is put after non-linearity

### BN and activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| ReLU |0.499| 2.21 | |
| RReLU |0.500| 2.20 | |
| PReLU |**0.503**| **2.19** | |
| ELU |0.498| 2.23 | |
| Maxout |0.487| 2.28| |
| Sigmoid |0.475| 2.35| |
| TanH |0.448| 2.50 | |
| No |0.384| 2.96 | |

### BN and dropout

ReLU non-linearity, fc6 and fc7 layer only

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
|  Dropout = 0.5 |0.499| 2.21 |  |
|  Dropout = 0.2 |**0.527**| **2.09** |  |
|  Dropout = 0 |0.513| 2.19 | |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs)

### BN-arch-init
| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Caffenet |0.471| 2.36 ||
| Caffenet BN Before + scale&bias layer LSUV|0.478| 2.33 | |
| Caffenet BN Before + scale&bias layer Ortho|0.482| 2.31 | |
| Caffenet BN After LSUV | 0.499 | 2.21 | |
| Caffenet BN After Ortho | **0.500** | **2.20** | |

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [GoogLeNet128](http://arxiv.org/abs/1409.4842) | **0.619** | **1.61** ||
| GoogLeNet BN Before + scale&bias layer LSUV||  | In progress|
| GoogLeNet BN Before + scale&bias layer Ortho| 0.607| 1.67 | |
| GoogLeNet BN After LSUV |0.596| 1.70 | |
| GoogLeNet BN After Ortho |0.584 | 1.77  | |

    
[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/bn_init), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/bn_init)

### Batch size, ReLU

Tanh results are moved [here] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/BatchSize.md)

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

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/batch_size), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/batch_size)

So general recommendation: too big batch_sizes leads to a bit inferior results, but in general batch_size should be selected based computation speed. If learning rate is adjusted, than no practial differenc
e between different batch sizes.

### From contributors

Base net is caffenet+BN+ReLU+drop=0.2
There difference in filters (main, 5x5 -> 3x3 + 3x3 or 1x5+5x1) and solver.

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
|  Base |0.527| 2.09 |  |
|  Base_dereyly_lr, noBN, ReLU |0.441| 2.53 | max_iter=160K, stepsize=2K, gamma=0.915, but default caffenet|
|  Base_dereyly 5x1, noBN, ReLU|0.474| 2.31 | 5x5->1x5+5x1 |
|  Base_dereyly_PReLU |0.550| 1.93 | BN, PreLU + base_lr=0.035, exp lr_policy, 160K iters, 5x5->3x3+3x3 |
|  Base_dereyly 3x1|**0.553**| **1.92** | PreLU + base_lr=0.035, exp lr_policy, 160K iters, 5x5->1x3+1x3+3x1+1x3 |
|  Base_dereyly 3x1 scale aug|0.530| 2.04 | Same as previous, img: 128 crop from (128...300)px image, test resize to 144, crop 128 |
|  Base_dereyly 3x1 scale aug|0.512| 2.17|  Same as previous, img: 128 crop from (128...300)px image, test resize to (128+300)/2, crop 128 |
|  Base_dereyly 3x1->5x1|0.546| 1.97* | PreLU + base_lr=0.035, exp lr_policy, 160K iters, 5x5->1x5+1x5+5x1+1x5 |
|  Base_dereyly 3x1,halfBN|0.544| 1.95 | PreLU + base_lr=0.035, exp lr_policy, 160K iters,5x5->1x3+1x3+3x1+1x3, BN only for pool and fc6 |
|  Base_dereyly 5x1|0.540| 2.00 | PreLU + base_lr=0.035, exp lr_policy, 160K iters, 5x5->1x5+5x1 |
|  DarkNetBN |0.502| 2.25 | 16C3->MP2->32C3->MP2->64C3->MP2->128C3->MP2->256C3->MP2->512C3->MP2->1024C3->1000CLF.BN|
| | | | + PreLU + base_lr=0.035, exp lr_policy, 160K iters |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/contrib), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/contrib)

### Network width

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| 2 wider| **0.533** | **2.04** | |
| sqrt(2) wider | 0.506 | 2.17 | |
| Default | 0.471 | 2.36 | |
| sqrt(2) narrower | 0.460 | 2.41 | |
| 2x narrower |  0.416 | 2.68 | |


[logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/contrib/complexity)

### Dataset size

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| Default, 1.2M images | **0.471** | **2.36** | |
| 800K images |0.438| 2.54 | |
| 600K images |0.425| 2.63 | |
| 400K images |0.393| 2.92 | |
| 200K images |0.305| 4.04 | |


### Dataset size, no RGB scaling 
Or why input var=1 for LSUV is so important

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| 800K images |0.438| 2.54 | |
| 600K images |0.425| 2.63 | |
| 600K images, no scale |0.379| 2.92 | |
| 400K images |0.393| 2.92 | |
| 400K images, no scale |0.357| 3.10 | |
| 200K images |0.305| 4.04 | |
| 200K images, no scale |0.277| 4.06 | |

[logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/contrib)

### Other

ReLU non-linearity, fc6 and fc7 layer only

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
|  Default |0.471| 2.36 ||
|  NoBias |0.445| 2.50 | Biases initialized with zeros, lr_rate = 0|

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/other), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/other)


The PRs with test are welcomed

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Anything vs. seconds" will give weird results. 

