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
epoches = 50
batch_size = 100
alpha = 1
record_interval = 100
'''
END
'''

train_data, test_data = MNIST_Dataset()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model = VAE().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
# recon_loss_fn = torch.nn.MSELoss(reduce='sum')
recon_loss_fn = lambda recon_x, x: F.binary_cross_entropy(recon_x, x, reduction='sum') 
kl_loss_fn = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss_record, kl_loss_record = [], []
for epoch in range(epoches):
    recon_loss_batch, kl_loss_batch = 0, 0
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        gen_imgs, mu, logvar = model(images)
        recon_loss = recon_loss_fn(gen_imgs, images)
        kl_loss = kl_loss_fn(mu, logvar)
        loss = recon_loss + alpha * kl_loss 

        optim.zero_grad()
        loss.backward()
        optim.step()

        recon_loss_batch += recon_loss.item()
        kl_loss_batch += kl_loss.item()
        if (i + 1) % record_interval == 0:
            print(f"Epoch [{epoch+1}/{epoches}], Batch [{i+1}/{len(train_loader)}], Recon Loss: {recon_loss_batch/(record_interval*batch_size):.4f}, KL Loss: {kl_loss_batch/(record_interval*batch_size):.4f}")
            recon_loss_record.append(recon_loss_batch/100)
            kl_loss_record.append(kl_loss_batch/100)
            recon_loss_batch, kl_loss_batch = 0, 0
    
    # Test
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            gen_imgs, mu, logvar = model(images)
            recon_loss = recon_loss_fn(gen_imgs, images)
            kl_loss = kl_loss_fn(mu, logvar)
            test_loss += (recon_loss + alpha * kl_loss).item()
        test_loss /= len(test_loader.dataset)
        print(f"Test Loss: {test_loss:.4f}")
        


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
