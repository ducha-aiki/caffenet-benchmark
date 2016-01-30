This is quick evaluation of different architectures functions performance on ImageNet-2012. 

The architecture is similar to common ones for ImageNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)

### Architectures

### Architectures

CaffeNet only

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [CaffeNet256](http://arxiv.org/abs/1408.5093) | **0.565** | **1.87** | Reference BVLC model, LSUV init|
| [CaffeNet](http://arxiv.org/abs/1408.5093) |0.470| 2.36 | Pool5 = 3x3|
| CaffeNetAll | **0.530** | **2.05** | All improvements without caffenet arch change: ELU + SPP + color_trans3-10-3 + Nesterov+ (AVE+MAX) Pool + linear lr_policy|
|  | + 0.06 |  | Gain over vanilla caffenet128. "Sum of gains" = 0.018 + 0.013 + 0.015 + 0.003 + 0.013 + 0.023 = 0.085 |
| CaffeNet256All | 0.613 | 1.64 | All improvements without caffenet arch change: ELU + SPP + color_trans3-10-3 + Nesterov+ (AVE+MAX) Pool + linear lr_policy|
| CaffeNet, no pad |0.411 | 2.70 | No padding, but conv1 stride=2 instead of 4 to keep size of pool5 the same|
| CaffeNet, dropout in conv | 0.426| 2.60 | Dropout before pool2=0.1, after conv3 = 0.1, after conv4 = 0.2 |
| [CaffeNetSPP](http://arxiv.org/abs/1406.4729) |0.483| 2.30 | SPP= 3x3 + 2x2 + 1x1 |
| [DarkNetBN](http://pjreddie.com/darknet/imagenet/#reference) |0.502| 2.25 | 16C3->MP2->32C3->MP2->64C3->MP2->128C3->MP2->256C3->MP2->512C3->MP2->1024C3->1000CLF.BN|
| | | | + PreLU + base_lr=0.035, exp lr_policy, 160K iters |

Others

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [DarkNetBN](http://pjreddie.com/darknet/imagenet/#reference) |0.502| 2.25 | 16C3->MP2->32C3->MP2->64C3->MP2->128C3->MP2->256C3->MP2->512C3->MP2->1024C3->1000CLF.BN|
| [HeNet2x2](http://arxiv.org/abs/1412.1710) |0.561 | 1.88|No SPP, Pool5 = 3x3, VLReLU, J' from paper |
| [HeNet3x1](http://arxiv.org/abs/1412.1710) |0.560 | 1.88|No SPP, Pool5 = 3x3, VLReLU, J' from paper, 2x2->3x1 |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.407| 2.80| With BN, huge overfitting, no dropout. First attempt to train, probably smth went wrong |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.518| 2.16| Without BN, less overfitting (still > than HeNet), no dropout. Looks like needs bigger initial LR, than caffenet. 3rd attempt is on-going  |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.525| 2.12| Without BN, less overfitting (still > than HeNet), no dropout. LR=0.05 until 25K iters.  |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.567| 1.91| Without BN, ELU, linear lr_policy.  |
| [GoogLeNet](http://arxiv.org/abs/1409.4842) | **0.619** | **1.61** |linear lr_policy, batch_size=256. obviously slower than caffenet |
| googlenet_loss2_clf| 0.571 | 1.80 | from net above, aux classifier after inception_4d |
| googlenet_loss1_clf| 0.520 | 2.06 | from net above, aux classifier after inception_4a |


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/architectures), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/architectures)

Architectures tested:

1. CaffeNet (pool5 size = 3x3)
2. HeNet [Convolutional Neural Networks at Constrained Time Cost](http://arxiv.org/abs/1412.1710). The difference with paper is VLReLU (converges faster at start) and no SPP pooling, instead used "classical" pool5
3. CaffeNetSPP, single scale training (SPP pool5 = 3x3 + 2x2 + 1x1) [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](http://arxiv.org/abs/1406.4729)]
4. ThinResNet-101 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) with BN, 1st attempt
5. ThinResNet-101 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) without BN, less stride in first layer but thinner than (4), 2nd attempt
6. ThinResNet-101 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) as 5, but lr=0.05 until 25K iterations.
7. ThinResNet-101 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) as 5, but with ELU and lr_policy=linear.
8. GoogleNet [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)

![CaffeNet128 test accuracy](/logs/architectures/img/0.png)
![CaffeNet128 test loss](/logs/architectures/img/2.png)
![CaffeNet128 train loss](/logs/architectures/img/6.png)

Architectures are selected, that their theoretical and/or practical computational complexity ~ caffenet. Currently, holds for all except HeNet, which is slower in practice. 


*** Contib 
Base net here is caffenet+BN+PReLU+dropout=0.2


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



The PRs with test are welcomed

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/contrib), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/contrib)


![CaffeNet128 test accuracy](/logs/contrib/img/0.png)
![CaffeNet128 test loss](/logs/contrib/img/2.png)
![CaffeNet128 lr_rate](/logs/contrib/img/2.png)
![CaffeNet128 train loss](/logs/contrib/img/6.png)


P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

