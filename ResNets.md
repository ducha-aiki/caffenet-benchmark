This is quick error-and-trial ResNet training attempts ImageNet-2012. 

The architecture is similar to common ones for ImageNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)

### Architectures, bad, overfitting

Except GoogLeNet


| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.407| 2.80| 3x thinner, than in paperWith BN, huge overfitting, no dropout. First attempt to train, probably smth went wrong |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.518| 2.16| Without BN, less overfitting (still > than HeNet), no dropout. Looks like needs bigger initial LR, than caffenet. 3rd attempt is on-going  |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.525| 2.12| Without BN, less overfitting (still > than HeNet), no dropout. LR=0.05 until 25K iters.  |
| [ThinResNet-101](http://arxiv.org/abs/1512.03385) | 0.567| 1.91| Without BN, ELU, linear lr_policy.  |
| [ResNet-56BN](http://arxiv.org/abs/1512.03385) | 0.612| 1.70| With BN, no EA, step lr_policy. Stopped at 220K iters due to overfitting  |
| [ResNet-56ELU bad :(](http://arxiv.org/abs/1512.03385) | 0.515| 2.43| Double stride after maxpool1 by mistake :( Without BN, ELU, step lr_policy. Stopped at 220K iters due to overfitting  |
| DereResNet-45 | 0.495| 2.25| Double stride after maxpool1 by mistake :(  With BN, no EA, dereyly lr_policy. Stopped at 120K iters due to overfitting  |
| [GoogLeNet-128](http://arxiv.org/abs/1409.4842) | **0.619** | **1.61** | For reference. linear lr_policy, batch_size=256. |

### Architectures, good attempts


| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [ResNet-50ELU-2xThinner](http://arxiv.org/abs/1512.03385) | 0.616| 1.63| Without BN, ELU, dropout=0.2 before classifier. 2x thinner, than in paper. Quite fast. No large overfitting (unlike upper table) |
| GoogLeNet128Res | **0.634** | **1.56** |linear lr_policy, batch_size=256. Resudial connections between inception block. No BN |
| [GoogLeNet-128](http://arxiv.org/abs/1409.4842) | 0.619 | 1.61 | For reference. linear lr_policy, batch_size=256. |
| VggLikeResNet-50-ELU-RoR-var | 0.626| 1.59| Step LR policy, max_iter = 200K, no BN, 4x thinner than VGG, [Residual on residual](https://arxiv.org/abs/1608.02908)  .|
| VggLikeResNet-50-ELU | 0.632| 1.57| Step LR policy, max_iter = 200K, no BN, 4x thinner than VGG. More RoR .|
| VggLikeResNet-50-ELU-RoR 1x5 | 0.628| 1.58| Step LR policy, max_iter = 200K, no BN, 4x thinner than VGG. 1x5 layers|
| VggLikeResNet-50-ELU-RoR 1x3 | 0.631 | 1.58| Step LR policy, max_iter = 200K, no BN, 4x thinner than VGG . |



![ResNet128 test accuracy](/logs/architectures/resnet/img/0.png)


![ResNet128 test loss](/logs/architectures/resnet/img/2.png)


![ResNet128 train loss](/logs/architectures/resnet/img/6.png)

