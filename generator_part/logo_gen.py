import torch
import numpy as np
import random


from .models import Generator
from .wgan import GoodGenerator32

weight_path= "gen_logo_model_wgan.pt"


netG = GoodGenerator32(dim=64, latent_dim = 128,  output_dim=3*32*32)
netG.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

netG.eval()

def gen_logo_color(noise = None):
    
    """
    Принимает на вход вектор шума либо генерит случайный
    затем генерирует картинку 32х32 и возвращает ее.
    Модель сделана с помощью iWGAN-GP
    
    """
    
    # netG.cuda()

    if not noise:
        # noise = torch.randn(1, 128, device="cuda")
        noise = torch.randn(1, 128)

    with torch.no_grad():
        # Get generated image from the noise vector using
        # the trained generator.
        generated_img = netG(noise).detach().cpu()

    return generated_img[0].cpu().detach().numpy().transpose(1,2,0)
