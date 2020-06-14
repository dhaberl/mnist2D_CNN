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
Download the dataset (here: 2D MNIST as npy-files) from: [LINK] 

or use your own (shape of npy-files: 28x28x1). 

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

### Train and test
Set data directory and define hyperparameters, e.g.:
```
- data_dir = 'data/'
- num_epochs = 50
- batch_size = 32
- train_ratio = 0.7
- validation_ratio = 0.15
- test_ratio = 0.15
```

Run:
```
python main.py
```

## Acknowledgments
- The code of the Data Generator is based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
