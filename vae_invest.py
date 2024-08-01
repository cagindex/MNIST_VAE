import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import CVAE
from utils import vae_path, MNIST_Dataset, device
from IPython import embed

model = CVAE().to(device)
model.load_state_dict(torch.load(vae_path, weights_only=True))


train_data, test_data= MNIST_Dataset()


def sample_image(idx):
    """采样并展示对比原图和生成图
    :param int idx: 训练集中的图像下标
    """
    img, label = train_data[idx]
    with torch.no_grad():
        gen_img, _, _ = model(img.unsqueeze(0).to(device), torch.tensor([label]).to(device))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img.numpy().squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(gen_img.detach().cpu().numpy().squeeze(), cmap='gray')
    plt.title('Generated Image')
    plt.show()


def latent_space_data_collect():
    print('Collecting latent space data...')
    ret_value = { i : [] for i in range(10)}
    with torch.no_grad():
        for img, label in tqdm(train_data):
            latent_vector = model.encode(img.unsqueeze(0).to(device), torch.tensor([label]).to(device))
            mu, logvar = latent_vector.chunk(2, dim=1) 
            ret_value[label].append(mu.detach().cpu().numpy().flatten())
    return ret_value


def latent_space_visualization(latent_space_data = None):
    if latent_space_data is None:
        latent_space_data = latent_space_data_collect()
    latent_vector_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i in range(10):
        latent_vectors = latent_space_data[i]
        plt.scatter([v[0] for v in latent_vectors], [v[1] for v in latent_vectors], c=latent_vector_colors[i], label=str(i), s=5)
    plt.legend()
    plt.show()


def latent_space_generation(latent_vector, label):
    latent_vector = torch.cat([latent_vector.reshape(1, -1), torch.tensor(label).reshape(1, -1)], dim=1)
    gen_img = model.decode(torch.tensor(latent_vector).to(device))
    plt.imshow(gen_img.detach().cpu().numpy().squeeze(), cmap='gray')
    plt.title('Generated Image')
    plt.show()


def latent_vector_pca(latent_space_data):
    all_data = []
    for key, value in latent_space_data.items():
        all_data += value
    all_data = np.array(all_data)
    pca = PCA(n_components=2)
    pca.fit(all_data)
    latent_vectors = pca.transform(all_data)

    res = {}
    start_idx = 0
    for key, value in latent_space_data.items():
        end_idx = start_idx + len(value)
        res[key] = latent_vectors[start_idx: end_idx]
        start_idx = end_idx
    return res


def visualize_latent_space_pca():
    latent_data = latent_space_data_collect()
    res = latent_vector_pca(latent_data)
    latent_space_visualization(res)


# latent_space_visualization()
# latent_space_generation([0.5, 0.5])
# sample_image(100)
embed()

