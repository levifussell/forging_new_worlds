## Forging new worlds
### High-resolution synthetic galaxies with chained generative adversarial networks

This repository contains supplementary code for the paper _Forging new worlds: high-resolution synthetic galaxies with chained generative adversarial networks_, a pre-print version which can be found on arXiv ([arXiv:1811.03081](https://arxiv.org/abs/1811.03081)).

At the moment, this repository only provides the trained models for generating galaxies using our chained GAN model. The code necessary for training the models from scratch with your own galaxy datasets will be available soon.

#### File description

#### Quickstart guide

To generate galaxies (in either Python2.x or Python3.x) use the following:

```
python generate_galaxies.py
```

The only potentially useful argument to change is the batchsize:

```
python generate_galaxies.py --batchSize 16
```
