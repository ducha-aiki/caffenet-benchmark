This is quick evaluation of different colorspace preprocessing performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

### Colorspace

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| RGB |0.470| 2.36 | default, no changes. Input = 0.04 * (Img - [104, 117,124]) |
| RGB_by_BN |0.469| 2.38 | Input = BatchNorm(Img)|
| CLAHE| 0.467 | 2.38 | RGB -> LAB -> CLAHE(L)->RGB->BatchNorm(RGB) |
| YCrCb | 0.458| 2.42 | RGB->YCrCb->BatchNorm(YCrCb)  |
| HSV |0.451| 2.46 |RGB->HSV->BatchNorm(HSV)  |
| Lab |-| - | Doesn`t leave 6.90 loss after 1.5K iters |
| RGB->10->3 TanH | 0.463| 2.40 | RGB -> conv1x1x10 tanh  ->  conv1x1x3 tanh |
| RGB->10->3 VlReLU | **0.485** | **2.28** | RGB -> conv1x1x10 vlrelu  ->  conv1x1x3 vlrelu|
| RGB and log(RGB)->10->3 VlReLU | 0.482 | 0.482 | RGB and log (RGB) -> conv1x1x10 vlrelu  ->  conv1x1x3 vlrelu|
| NN-Scale |0.467| 2.38 | Nearest neightbor instead of linear interpolation for rescale. Faster, but worse :(|


[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/colorspace), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/colorspace)



![CaffeNet128 test accuracy](/logs/colorspace/img/0.png)


![CaffeNet128 test loss](/logs/colorspace/img/2.png)


![CaffeNet128 train loss](/logs/colorspace/img/6.png)

