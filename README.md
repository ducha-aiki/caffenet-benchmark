Welcome to evaluation of CNN design choises performance on ImageNet-2012. Here you can find prototxt`s of tested nets and full train logs.

The basic architecture is similar to CaffeNet, but has several differences:

1. Images are resized to small side = 128 for speed reasons. Therefore pool5 spatial size is 3x3 instead of 6x6.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)
4. Because LRN layers add nothing to accuracy (validated [here](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)), they were removed for speed reasons in most experiments.

On-going evaluations with graphs:
- [activations](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Activations.md)
- [architectures] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Architectures.md)
- [augmentation] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/Augmentation.md)
- [batchnorm] (https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)

The PRs with test are welcomed

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Anything vs. seconds" will give weird results. 

