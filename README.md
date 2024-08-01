# VAE

- `vae_train.py` is the main file to train the VAE model.
- `vae_utils.py` contains the helper functions used in `vae_train.py`.
- `vae_model.py` contains the VAE model architecture.
- `vae_invest.py` using `IPython.embed()` to check the model's output.

## vae_invest.py

- `sample_image(idx)` given an idx, it shows the original image and the reconstructed image.

- `latent_space_data_collect()` collects the latent space data for all the images in the train dataset.

- `latent_space_visualization()` plots the latent space data.(**The latent space should be 2D for visualization**)

- `latent_space_generation(latent_vector)` given a latent vector, it generates a new image.

- `latent_vector_pca(latent_space_data)` If the latent space is high-dimensional, it can be reduced to 2D using PCA, and return the reduced data which can be passed to `latent_space_visualization()`.

- `visualize_latent_space_pca()` plots the reduced latent space data using PCA. It is a wrapper function for `latent_space_data_collect()` and `latent_vector_pca()` and `latent_space_visualization()`.
