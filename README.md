# Convolutional Neural Network in Keras/Tensorflow
 Classification CNN for 2D MNIST data.

## Prerequisites
- Linux or Windows 
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3
- env_mnist2d_cnn.yml

## Getting Started
### Branches
- master: standard implementation of the CNN
- DataGenerator2D: implementation of the CNN using a custom data generator and data augmentation.

### Installation
- Clone or download this repo
- Install dependencies (see env_mnist2d_cnn.yml) and set up your environment

### Dataset
A subset of 42 000 grey-scale images of the original MNIST database was used. Each image contains 28x28 pixels, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating the brightness (low values) or darkness (high values) of that pixel. This pixel-value is an integer between 0 (white) and 255 (black). 

The images are stored as npy-files. The dataset also contains a csv-file with the ID and the corresponding ground truth label.

Download the dataset from: [LINK] 

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
img_1; 7 \
...

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
- The organization of the dataset is based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
