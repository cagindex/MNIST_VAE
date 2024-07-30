import torch
import torch.nn as nn
from utils import device


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 4),
        ) 

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Unflatten(1, (1, 28, 28)),
            nn.Sigmoid(),
        )
    
    def reparameterize(self, mu, logvar):
        z = torch.normal(0, 1, size=logvar.size(), requires_grad=False).to(device)
        return mu + (0.5*logvar).exp() * z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # 必须取logvar，如果取sigma的话klloss中取 log 后可能出现nan
        latent_vector = self.encode(x)
        mu, logvar = latent_vector.chunk(2, dim=1) 
        z = self.reparameterize(mu, logvar)
        gen_img = self.decode(z)
        return gen_img, mu, logvar

