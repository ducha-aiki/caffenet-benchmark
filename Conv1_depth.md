This is quick evaluation of conv1 depth impact on performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Default augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.
Here I am trying to check how much improvement can we get by learning "deeper" == "more complex" conv1, without receptive field change.
To do this, I am adding series of 96 channels 1x1 convolution + ReLU after conv1, before pool1. 

For comparison I also add 96 channels 3x3 convolution, which has more parameters AND - which is more important - which increases receptive field


### Conv1 depth

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| ------:|:-----------|
| Default, no 1x1 or 3x3 |  0.471 | 2.36 | conv1 -> pool1 |
| + 1x1x96 NiN |  0.490 | 2.24 | conv1 -> 96C1 -> pool1 |
| + 3x (1x1x96 NiN) |  0.509 | 2.10 | conv1 -> 3x(96C1) -> pool1 |
| + 5x (1x1x96 NiN) |  0.514 | 2.11 | conv1 -> 5x(96C1) -> pool1 |
| + 7x (1x1x96 NiN) |  0.514 | 2.11 | conv1 -> 7x(96C1) -> pool1 |
| + 9x (1x1x96 NiN) |  0.516 | 2.10 | conv1 -> 9x(96C1) -> pool1 |
| + 9x (1x1x96 NiN)R |  0.509 | 2.13 | conv1 -> Residual9x(96C1) -> pool1. 276k iters |
| + 1x (3x3x96 NiN) |  0.500 | 2.19 | conv1 -> 1x(96C3) -> pool1 |
| + 3x (3x3x96 NiN) |  0.538 | 1.99 | conv1 -> 1x(96C3) -> pool1 |
| + 5x (3x3x96 NiN) |  **0.551** | **1.91** | conv1 -> 1x(96C3) -> pool1 |

So impact of "more complex conv1" without inreasing the receptual field quickly saturates, while for 3x3 convolutions - not. 
See prototxt in the beginning of the log file [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/conv1_depth)


![CaffeNet128 test accuracy](/logs/conv1_depth/img/0.png)

![CaffeNet128 test accuracy](/logs/conv1_depth/img/0_3x3.png)

![CaffeNet128 test accuracy](/logs/conv1_depth/img/0_all.png)
