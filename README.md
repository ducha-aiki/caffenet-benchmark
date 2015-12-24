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

### Activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [ReLU](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf) |0.470:| 2.36 :|: With LRN layers|
| ReLU |0.470:| 2.36 :|: No LRN, as in rest |
| [PReLU](http://arxiv.org/abs/1502.01852) |:0.485:| 2.29 :|: ---------|
| [VLReLU](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf) |:0.469:| 2.40:|: y=max(x,x/3)|
| [RReLU](http://arxiv.org/abs/1505.00853) |:0.478:| 2.32:|: ---------|
| [ELU](http://arxiv.org/abs/1511.07289) |:0.488:| 2.28:|: ---------|
| [Maxout](http://arxiv.org/abs/1302.4389) |:0.482:| 2.30:|: sqrt(2) narrower layers|
[Prototxt](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/prototxt/activations), [logs](https://github.com/ducha-aiki/caffenet-benchmark/tree/master/logs/activations)


The PRs with test are welcomed

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Anything vs. seconds" will give weird results. 

