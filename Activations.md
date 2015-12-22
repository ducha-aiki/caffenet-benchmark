This is quick evaluation of different activation functions performance on ImageNet-2012. 

The architecture is similar to CaffeNet, but has differences:

1. Images are resized to small side = 128 for speed reasons.
2. fc6 and fc7 layers have 2048 neurons instead of 4096. 
3. Networks are initialized with [LSUV-init](http://arxiv.org/abs/1511.06422)

Because LRN layers add nothing to accuracy, they were removed for speed reasons in further experiments.
*ELU curves is unsmooth because of incorrectly set test set size. However, results from 310K to 320K iterations are obtained with fixed set size

![CaffeNet128 test accuracy](/logs/activations/img/0.png)


![CaffeNet128 test loss](/logs/activations/img/2.png)


![CaffeNet128 train loss](/logs/activations/img/6.png)

Previous results on small datasets like CIFAR (see [LSUV-init, Table3](http://arxiv.org/abs/1511.06422)) looks a bit contradictory to ImageNet ones so far.

Maxout net has two linear pieces and each has sqrt(2) less parameters than *ReLU networks, so overall complexity is same.

P.S. Logs are merged from lots of "save-resume", because were trained at nights, so plot "Accuracy vs. seconds" will give weird results. 

