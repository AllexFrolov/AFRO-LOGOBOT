import os
import torch
import random
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import utils
from skimage import io
import matplotlib.pyplot as plt

from model import Generator


pics = 1
sample = 1
size = 32
truncation = 0.5
truncation_mean = 4096
ckpt = "icons_gen.pt"
#ckpt = "../networks/torch_networks/stylegan2-ffhq-config-f.pt"
latent = 512
n_mlp = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GenModel(object):

    def __init__(self):
        super().__init__()
        self.backbone = Generator(size, latent, n_mlp, channel_multiplier=2)
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(checkpoint['g_ema'])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.to(device)
        self.mean_latent = self.backbone.mean_latent(truncation_mean)
        
    def generate_logo(self):
        sample_z = torch.randn(sample, latent, device=device)
        x = self.backbone([sample_z], truncation=truncation, truncation_latent=self.mean_latent)
        
        return x[0].cpu().detach().numpy().squeeze().transpose(1,2,0)
    

model = GenModel()
model.generate_logo()