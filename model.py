import torch
import torch.nn as nn
from utils import device


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # 6 @ 14x14
            nn.Conv2d(6, 16, 3, padding=1), # 16 @ 14x14
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # 16 @ 7x7
            nn.Flatten(),
            nn.Linear(16*7*7, 512), # 512
        ) 

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1 * 28 * 28), # 1 * 28 * 28
            nn.Unflatten(1, (1, 28, 28)),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent_vector = self.encode(x)
        # 必须取logvar，如果取sigma的话klloss中取 log 后可能出现nan
        mu, logvar = latent_vector.chunk(2, dim=1) 
        sample = torch.normal(0, 1, size=logvar.size(), requires_grad=False).to(device)
        z = (mu + (0.5*logvar).exp() * sample)
        gen_img = self.decode(z)
        return gen_img, mu, logvar
