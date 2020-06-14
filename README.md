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

### Data Generator and Data Augmentation
The Data Generator generates the dataset in batches on multiple cores for real-time data feeding to the machine learning model. 

It can be used by importing it in the main file:

```
from DataGenerator import DataGenerator
```

and takes as an input:

- data_dir: path to the data directory
- list_ids: list of IDs as shown above
- labels: list of labels as shown above
- batch_size 
- dim: dimensions of the data (28x28)
- n_channels: number of channels
- n_classes: number of classes
- shuffle: whether to shuffle at generation or not (boolean) 
- **da_parameters

Data augmentation:

The Data Generator also allows real-time data augmentation.

```
da_parameters = {"width_shift": 5.,
                 "height_shift": 5.,
                 "rotation_range": 15.,
                 "horizontal_flip": 0.5,
                 "vertical_flip": 0.5,
                 "min_zoom": 0.7,
                 "max_zoom": 1.1,
                 "random_crop_size": 0.85,
                 "random_crop_rate": 1.,
                 "center_crop_size": 0.85,
                 "center_crop_rate": 1.,
                 "gaussian_filter_std": 1.,
                 "gaussian_filter_rate": 1.
                 }
```

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
