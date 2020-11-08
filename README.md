# 2D Convolutional Neural Network in Keras/Tensorflow
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

Download the dataset from: https://www.kaggle.com/c/digit-recognizer/data

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

### Data Generator
The Data Generator generates the dataset in batches on multiple cores for real-time data feeding to the machine learning model. 

The generator can be used by importing it in the main file:

```
from DataGenerator import DataGenerator
```

Input parameters are:

- data_dir: path to the data directory (string)
- list_ids: list of IDs as shown above (list)
- labels: list of labels as shown above (list)
- batch_size: number of samples that will be propagated through the network (integer)
- dim: dimensions of the data (tuple with intergers). E.g., image with 28x28 pixels => (28, 28)
- n_channels: number of channels (integer). E.g., RGB = 3 channels
- n_classes: number of classes (integer)
- shuffle: whether to shuffle at generation or not (boolean) 
- **da_parameters

### Data augmentation

The Data Generator also allows real-time data augmentation.

Augmentations:
- width_shift: Shifts are randomly sampled from [-width_shift, +width_shift].
- height_shift: Shifts are randomly sampled from [-height_shift, +height_shift].
- rotation_range: Degree range for random rotations. Randomly sampled from [-rotation_range, +rotation_range].
- horizontal_flip: Probability rate for horizontal flips.
- vertical_flip: Probability rate for vertical flips.
- min_zoom: Lower limit for a random zoom.
- max_zoom: Upper limit for a random zoom. The zoom factor is randomly sampled from [min_zoom, max_zoom].
- random_crop_size: Fraction of the total width/height. The final crop is performed by randomly sampling a section from the original image.
- random_crop_rate: Probability rate for random cropping.
- center_crop_size: Fraction of the total width/height. The final crop is based on the center of the image.
- center_crop_rate: Probability rate for centered cropping.
- gaussian_filter_std: Images are blurred by a Gaussian function which is defined by its standard deviation (std). The std is randomly sampled from [0, gaussian_filter_std].
- gaussian_filter_rate: Probability rate for gaussian filtering.

For example:

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
- [1] The organization of the dataset and the code of the Data Generator is based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
