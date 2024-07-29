import torch
import torch.nn as nn
from utils import device


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
        ) 

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1 * 28 * 28), # 1 * 28 * 28
            nn.Unflatten(1, (1, 28, 28)),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, mu, logvar):
        sample = torch.normal(0, 1, size=logvar.size(), requires_grad=False).to(device)
        z = (mu + (0.5*logvar).exp() * sample)
        return self.decoder(z)

    def forward(self, x):
        latent_vector = self.encode(x)
        # 必须取logvar，如果取sigma的话klloss中取 log 后可能出现nan
        mu, logvar = latent_vector.chunk(2, dim=1) 
        gen_img = self.decode(mu, logvar)
        return gen_img, mu, logvar
