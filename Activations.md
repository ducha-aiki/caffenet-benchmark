This is quick evaluation of different activation functions performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)

Because LRN layers add nothing to accuracy, they were removed for speed reasons in further experiments.
*ELU curves is unsmooth because of incorrectly set test set size. However, results from 310K to 320K iterations are obtained with fixed set size

### Activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| [ReLU](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf) |0.470| 2.36 | With LRN layers|
| ReLU |0.471| 2.36 | No LRN, as in rest |
| TanH |0.401| 2.78 |  |
| [VLReLU](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf) |0.469| 2.40|y=max(x,x/3)|
| [RReLU](http://arxiv.org/abs/1505.00853) |0.478| 2.32| |
| [Maxout](http://arxiv.org/abs/1302.4389) |0.482| 2.30| sqrt(2) narrower layers, 2 pieces|
| [PReLU](http://arxiv.org/abs/1502.01852) |0.485| 2.29 | |
| [ELU](http://arxiv.org/abs/1511.07289) |**0.488**| **2.28**| |
| Shifted [Softplus](http://machinelearning.wustl.edu/mlpapers/papers/AISTATS2011_GlorotBB11) |0.486| 2.29| Shifted BNLL aka softplus, y = log(1 + exp(x)) - log(2). Same as ELU, as expected |


### BN and activations

| Name    | Accuracy      | LogLoss | Comments  |
| -------|---------:| -------:|:-----------|
| ReLU |0.499| 2.21 | |
| RReLU |0.500| 2.20 | |
| PReLU |**0.503**| **2.19** | |
| ELU |0.498| 2.23 | |
| Maxout |0.487| 2.28| |
| Sigmoid |0.475| 2.35| |


![CaffeNet128 test accuracy](/logs/activations/img/0.png)


![CaffeNet128 test loss](/logs/activations/img/2.png)


![CaffeNet128 train loss](/logs/activations/img/6.png)

Previous results on small datasets like CIFAR (see [LSUV-init, Table3](http://arxiv.org/abs/1511.06422)) looks a bit contradictory to ImageNet ones so far.

Maxout net has two linear pieces and each has sqrt(2) less parameters than *ReLU networks, so overall complexity is same.

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

