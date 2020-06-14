# Convolutional Neural Network in Keras/Tensorflow
 Classification CNN for 2D MNIST data.

## Prerequisites
- Linux or Windows 
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3
- env_mnist2d_cnn.yml

## Getting Started
### Branches
### Installation
- Clone or download this repo
- Install dependencies (see env_mnist2d_cnn.yml) and set up your environment

### Dataset
Download the dataset (here: mnist 2D as NumPy files) from: [LINK] or use your own one (shape of npy-file should be: 28x28x1). 

folder/
- main.py
- DataGenerator.py
- data/
	- img_0.npy
	- ...
	- labels.csv

where labels.csv contains for instance:

ID; Label \
img_0; 2 \
. \
. \
.

### Run: Train and test the model
Set data directory and define hyperparameters, e.g.:
- data_dir = 'data/'
- num_epochs = 50
- batch_size = 32
- train_ratio = 0.7
- validation_ratio = 0.15
- test_ratio = 0.15

Run:

python main.py

## Acknowledgments
The notation/organization of the code is inspired by the blog post: \
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
