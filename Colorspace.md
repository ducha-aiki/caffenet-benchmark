This is quick evaluation of different colorspace preprocessing performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. No LRN layers.

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
| RGB->10->3 VlReLU | 0.485 | 2.28 | RGB -> conv1x1x10 vlrelu  ->  conv1x1x3 vlrelu|
| RGB->10->3 VlReLU->sum(RGB) | 0.482 | 2.30 | RGB -> conv1x1x10 vlrelu  ->  conv1x1x3 -> sum(RGB) ->vlrelu|
| RGB and log(RGB)->10->3 VlReLU | 0.482 | 2.29 | RGB and log (RGB) -> conv1x1x10 vlrelu  ->  conv1x1x3 vlrelu|
| RGB and log(RGB) and log (256-RGB)->10->3 VlReLU | 0.484 | 2.29 | RGB and log (RGB) and log (256 - RGB) -> conv1x1x10 vlrelu  ->  conv1x1x3 vlrelu|
| RGB->10->3 Maxout | **0.488** | **2.26** | RGB -> conv1x1x10 maxout(2)  ->  conv1x1x3 maxout(2)|
| RGB->16->3 VlReLU | 0.483 | 2.30 | RGB -> conv1x1x16 vlrelu  ->  conv1x1x3 vlrelu|
| RGB->3->3 VlReLU | 0.480 | 2.32 | RGB -> conv1x1x3 vlrelu  ->  conv1x1x3 vlrelu|
| NN-Scale |0.467| 2.38 | Nearest neightbor instead of linear interpolation for rescale. Faster, but worse :(|
| concat_rgb_each_pool |0.441| 2.51 | Concat avepoolRGB with each pool |
| OpenCV RGB2Gray |0.413| 2.70 |RGB->Grayscale Gray = 0.299 R + 0.587 G + 0.114 B |
| Learned RGB2Gray |0.419| 2.66 |RGB->conv1x1x1. -1.779 *R + 6.511 * G + 1.493 *B + 3.279  |

[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/colorspace), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/colorspace)



![CaffeNet128 test accuracy](/logs/colorspace/img/0.png)


![CaffeNet128 test loss](/logs/colorspace/img/2.png)


![CaffeNet128 train loss](/logs/colorspace/img/6.png)

