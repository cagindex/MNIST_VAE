import torch
import torch.nn as nn
from utils import device


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(785, 512),
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
            nn.Linear(3, 32),
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
    
    def encode(self, x, label):
        x = torch.cat([torch.flatten(x, start_dim=1), torch.reshape(label, (-1, 1))], dim=1)
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    # x shape: (batch_size, 1, 28, 28)
    # label shape: (batch_size, )
    def forward(self, x, label):
        latent_vector = self.encode(x, label)
        # 必须取logvar，如果取sigma的话klloss中取 log 后可能出现nan
        mu, logvar = latent_vector.chunk(2, dim=1) 
        z = self.reparameterize(mu, logvar) # z shape: (batch_size, 2)
        z = torch.cat([z, torch.reshape(label, (-1, 1))], dim=1)
        gen_img = self.decode(z)
        return gen_img, mu, logvar

