from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
#from torchvision import functional as F
import torch.functional as F

from torch.autograd import Variable
import numpy as np

import time
from collections import OrderedDict

from models import Generator_Stage1, Generator_Stage2

"""
Running this file will generate you a batch of galaxies by loading
 both GANs and feeding some 100-D noise through the pipeline. Feel
 free to use this code as a model for using this model.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dcgan_G', type=str, default='saved_models/dcgan_G.pth', help='filename of the 32->64 generator model')
parser.add_argument('--stackgan_G', type=str, default='saved_models/stackgan_G.pth', help='filename of the 64->128 generator model')
parser.add_argument('--batchSize', type=int, default=6, help='number of galaxy images to create')

opts = parser.parse_args()
print(opts) # print user choices

# fix for python 2 as per here: https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560/2
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# ---------------------------------------------------------------------------------------------------

def run_pipeline():
    # set seed
    r_seed = np.random.randint(1, 1000000)
    torch.manual_seed(r_seed)

    # load models --------------------------------------

    # load stage-I Generator
    dcgan_G = Generator_Stage1()
    dcgan_G.load_state_dict(torch.load(opts.dcgan_G, map_location=lambda storage, loc: storage))

    # load stage-II Generator
    stackgan_G = Generator_Stage2()
    st = torch.load(opts.stackgan_G, map_location=lambda storage, loc: storage)
    new_st = OrderedDict()
    for k,v in st.items():
        if 'num_batches_tracked' not in k:
            new_st[k] = v
    stackgan_G.load_state_dict(new_st)

    # feed noise through pipeline
    noise = Variable(torch.randn(opts.batchSize, 100, 1, 1))
    fake_64 = dcgan_G(noise)
    # dcgan produces [-1,1] images so we need to transform them to
    #  [0,1] for the Stage-II input.
    fake_64 += 1.0
    f_min = torch.min(fake_64)
    f_max = torch.max(fake_64)
    fake_64  -= f_min
    fake_64 /= f_max

    fake_128 = stackgan_G(fake_64)

    # save the images --------------------------------
    vutils.save_image(fake_128.data,
            'samples/fake_sample_{}.png'.format(r_seed),
            normalize=True)

if __name__ == "__main__":
    run_pipeline()
