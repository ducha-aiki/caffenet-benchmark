This is several tests that help to understand benefits 1xN+Nx1 architecture that approximate NxN+1x1.
Standard size is 1x3+3x1
Approximating just 3x3 without adding 1x1 is incorrect, because we change depth of the network.

I take good topology -- caffenet128_lsuv_no_lrn_BatchNormAfterPReLU.prototxt
Some changes:
- init is MSRA
- BN after pooling
- BN is one phase
- solver in different file
- solver has different parametrs reducing number of iterations

