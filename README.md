# tgan-pytorch
A PyTorch implementation of [Temporal Generative Adversarial Nets with Singular Value Clipping](https://arxiv.org/abs/1611.06624).

## Implementation details
Although in the original implementation Wasserstein GANs with weight clipping and singular value clipping were used,
this version uses the training procedure highlighted in [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).
The model is trained on MovingMNIST.
## Requirements
[PyTorch](http://pytorch.org/)  
[torchvision](https://github.com/pytorch/vision/tree/master/torchvision)  
[PyYAML](https://pypi.python.org/pypi/PyYAML)  

## Configuration
The configuration options are saved in _config.yml_. Modify its content accordingly to your needs, for example setting _use_cuda_ to `False`.

## Usage
Download the MovingMNIST dataset in the `data/` directory using:
```
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
```
To start the training:
```
python train.py
```
Savings will be contained in the `checkpoints/` directory, while images representing the generated videos are stored in `samples/`.

## Citation of original authors
I am not one of the authors of the original work nor affiliated with them. To reference their work please use:
```
@inproceedings{TGAN2017,
    author = {Saito, Masaki and Matsumoto, Eiichi and Saito, Shunta},
    title = {Temporal Generative Adversarial Nets with Singular Value Clipping},
    booktitle = {ICCV},
    year = {2017},
}
```
