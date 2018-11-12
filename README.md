## Forging new worlds
### High-resolution synthetic galaxies with chained generative adversarial networks

This repository contains supplementary code for the paper _Forging new worlds: high-resolution synthetic galaxies with chained generative adversarial networks_, a pre-print version of which can be found on arXiv ([arXiv:1811.03081](https://arxiv.org/abs/1811.03081)).

At the moment, this repository only provides the trained models for generating galaxies using our chained GAN model. The code necessary for training the models from scratch with your own galaxy datasets will be available soon.

#### File description

* `generate_galaxies.py`:

* `models.py`:

* `dcgan_G.pth`:

* `stackgan_G.pth`:

#### Quickstart guide

Generating galaxies with the provided pre-trained models works in both Python 2 and Python 3. In order to create a set of synthetic galaxy images, you can use the following command:

```
python generate_galaxies.py
```

If you require a specific number of synthetic samples, modify the `batchSize` parameter with that number:

```
python generate_galaxies.py --batchSize 16
```
