import torch
import random
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor

'''
parameters definition
'''
vae_path = './models/vae.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
functions definition
'''
def MNIST_Dataset():
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    return train_data, test_data

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
