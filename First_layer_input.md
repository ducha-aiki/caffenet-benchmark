This is quick evaluation of input and parameters of conv1 impact on performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

Default augmentation: random crop 128x128 from 144xN image, 50% random horizontal flip.

All the variants have similar computational complexity, if other not explicit said 

### Conv1 parameters

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| ------:|:-----------|
| Default, 128_K11_S4 |  **0.471** | 2.36 | Input size =128x128px, conv1 = 11x11x96, stride = 4 |
| 224_K11_S8 |0.453| 2.45 | Input size =256x256px, conv1 = 11x11x96, stride = 8. Not finished yet |
| 160_K11_S5 | 0.470| **2.35** | Input size =160x160px, conv1 = 11x11x96, stride = 5 |
| 96_K7_S3 | 0.459 | 2.43 | Input size =96x96px, conv1 = 7x7x96, stride = 3 |
| 64_K5_S2 | 0.445 | 2.50 | Input size =64x64px, conv1 = 5x5x96, stride = 2 |
| 32_K3_S1| 0.390 | 2.84 | Input size =32x32px, conv1 = 3x3x96, stride = 1 |
| 4x slower, 227_K11_S4 | 0.565 | 1.87 | Input size =227x227px, conv1 = 11x11x96, stride = 4,  Not finished yet |

[prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/conv1-input), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/conv1-input)

### Input image size

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| ------:|:-----------|


![CaffeNet128 test accuracy](/logs/conv1-input/img/0.png)


![CaffeNet128 test loss](/logs/conv1-input/img/2.png)


![CaffeNet128 train loss](/logs/conv1-input/img/6.png)

