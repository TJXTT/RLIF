# RLIF
Demo code of the paper "Relaxation LIF: A Gradient-based Spiking Neuron for Directly Training Deep Spiking Neural Networks"

## Run on CIFAR10
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_cifar10.py --data-path ../datasets/data_CIFAR10
```

## Run on DVS128-GESTURE
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_dvsgesture.py --data-path ../datasets/DVS128Gesture
```

