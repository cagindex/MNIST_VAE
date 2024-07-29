import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import VAE
from utils import MNIST_Dataset, seed_everything, vae_path, device

import matplotlib.pyplot as plt
from IPython import embed

seed_everything(2024)
'''
Hypter parameters
'''
lr = 0.001
epoches = 10
batch_size = 64
'''
END
'''

train_data, test_data = MNIST_Dataset()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model = VAE().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
# recon_loss_fn = torch.nn.MSELoss()
recon_loss_fn = lambda recon_x, x: F.binary_cross_entropy(recon_x, x, size_average=False) / x.shape[0]
kl_loss_fn = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu * mu - logvar.exp()) / mu.shape[0]
recon_loss_record = []
kl_loss_record = []
for epoch in range(epoches):
    recon_batch_loss = 0
    kl_batch_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        optim.zero_grad()
        gen_img, mu, logvar = model(images)
        recon_loss = recon_loss_fn(gen_img, images)
        kl_loss = kl_loss_fn(mu, logvar)
        loss = recon_loss + kl_loss
        loss.backward()
        optim.step()

        recon_batch_loss += recon_loss.item()
        kl_batch_loss += kl_loss.item()
        if (batch_idx+1) % 100 == 0:
            print(f"[Epoch {epoch+1}][Batch idx {batch_idx+1}], Avg Recon Loss: {recon_batch_loss/100:.4f}, Avg KL Loss: {kl_batch_loss/100:.4f}")
            recon_loss_record.append(recon_batch_loss/100)
            kl_loss_record.append(kl_batch_loss/100)
            recon_batch_loss, kl_batch_loss = 0, 0

torch.save(model.state_dict(), vae_path)
'''
plt
'''
x1 = [i for i in range(len(recon_loss_record))]
x2 = [i for i in range(len(kl_loss_record))]
plt.subplot(1,2,1)
plt.plot(x1, recon_loss_record)
plt.title("Recon Loss")
plt.subplot(1, 2, 2)
plt.plot(x2, kl_loss_record)
plt.title("KL Loss")
plt.show()
