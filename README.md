## Forging new worlds
### High-resolution synthetic galaxies with chained generative adversarial networks

This repository contains supplementary code for the paper _Forging new worlds: high-resolution synthetic galaxies with chained generative adversarial networks_, a pre-print version of which can be found on arXiv ([arXiv:1811.03081](https://arxiv.org/abs/1811.03081)).

At the moment, this repository mainly provides the trained models for generating galaxies using our chained GAN model. The code necessary for training the models from scratch with your own galaxy datasets will be available soon.

#### File description

* `generate_galaxies.py`: The user-facing Python file to create synthetic galaxy samples

* `models.py`: The DCGAN and StackGAN models used in the corresponding research and paper

* `dcgan_G.pth`: The pre-trained DCGAN generator used to create synthetic 64x64 pixel galaxy images

* `stackgan_G.pth`: The pre-trained StackGan generator used to upscale the DCGAN generator output

#### Quickstart guide

Generating galaxies with the provided pre-trained models works in both Python 2 and Python 3. In order to create one synthetic galaxy image, you can simply use the following command:

```shell
python generate_galaxies.py
```

If you want to create a specific number of synthetic samples, modify the `batchSize` parameter with that number:

```shell
python generate_galaxies.py --batchSize 16
```
