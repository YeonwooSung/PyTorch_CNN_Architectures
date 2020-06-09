# GoogLeNet

Implementation of the GoogLeNet, as introduced in the paper "Going Deeper with Convolutions" by Christian Szegedy et al. ([original paper](https://arxiv.org/abs/1409.4842))

## Prerequisites

- python >= 3.5
- pytorch==0.4.0
- numpy

You can install required packages by:

```bash
pip3 install -r requirements.txt
```

## DataSet

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

You could download the CIFAR-10 dataset by either visiting [this page](https://www.cs.toronto.edu/~kriz/cifar.html) or running the following scripts.

```bash
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

tar -xvzf cifar-10-python.tar.gz
```
