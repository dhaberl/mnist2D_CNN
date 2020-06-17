import os
import tensorflow as tf
from keras import Sequential, layers, regularizers
import keras
import numpy as np
import pandas as pd
import plotly.offline as ply
import plotly.graph_objs as graphs
import math
import random
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator

# Silent warning of Scipy.ndimage.zoom -> warning has no functional impact on the code
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

def get_dims(data_dir):
    """Get dimensions from a npy file in the given data directory data_dir."""
    npy_files = [file for file in os.listdir(data_dir) if file.endswith(".npy")]
    example_path = os.path.join(data_dir, npy_files[0])
    npy_example = np.load(example_path)

    return npy_example.shape


def load_data(ids, labels, data_dir):
    """Returns 2D image arrays and labels of given IDs. Note: This function is replaced by the Data Generator"""
    num_samples = len(ids)
    dims = get_dims(data_dir)
    X = np.empty((num_samples, dims[0], dims[1], dims[2]))   # Softcode dimensions
    y = np.empty(num_samples, dtype=int)

    # Fill data to X and y
    for i, ID in enumerate(ids):
        sample_path = os.path.join(data_dir, ID + '.npy')
        X[i, ] = np.load(sample_path).astype("float16")
        y[i] = labels[ID]

    return X, keras.utils.to_categorical(y)


def split_data(ids, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    """Split list of sample IDs randomly with a given ratio for the training, validation and test set."""

    # Check validity of ratio arguments
    if train_ratio + validation_ratio + test_ratio != 1.0:
        raise Exception("Error: train_ratio, validation_ratio and test_ratio must add up to 1.0")

    # Calculate number of samples in each set
    num_samples = len(ids)
    val_data_size = math.floor(num_samples * validation_ratio)
    test_data_size = math.floor(num_samples * test_ratio)
    train_data_size = num_samples - (val_data_size + test_data_size)

    # Randomize sample IDs
    random.seed(0)
    random.shuffle(ids)

    # Split data into training, validation and test set
    train_ids = ids[:train_data_size]
    val_ids = ids[train_data_size:train_data_size + val_data_size]
    test_ids = ids[train_data_size + val_data_size:]

    return {"train": train_ids, "validation": val_ids, "test": test_ids}


def plot_incorrects(model, test_images, test_labels, num_to_show=3):
    """Plot a number of incorrectly predicted images. Go to next image using Enter-key."""
    num_shown = 0

    for sample, label in zip(test_images, test_labels):
        if num_shown >= num_to_show:
            break
        prediction = predict(np.array([sample]), model, show=False)
        incorrects = prediction != [np.argmax(label)]
        if incorrects is True:
            print("True: {}\tPredicted: {}".format([np.argmax(label)], prediction))

            plt.imshow(sample[:, :, 0])
            plt.show()
            _ = input()     # Hit enter to go to next image

            num_shown += 1


def plot_train_val_acc(accs, show=True):
    """Plot training vs. testing accuracy over all epochs."""
    x = list(accs.keys())
    y_train = [i[0] for i in accs.values()]
    y_test = [i[1] for i in accs.values()]

    trace_train = graphs.Scatter(x=x, y=y_train, name="Training", mode="lines+markers",
                                 line=dict(width=4),
                                 marker=dict(symbol="circle",
                                             size=10))
    trace_test = graphs.Scatter(x=x, y=y_test, name="Validation", mode="lines+markers",
                                line=dict(width=4),
                                marker=dict(symbol="circle",
                                            size=10))

    layout = graphs.Layout(title="Training vs. Validation accuracy",
                           xaxis={"title": "Epoch"},
                           yaxis={"title": "Accuracy"})

    fig = graphs.Figure(data=[trace_train, trace_test], layout=layout)
    ply.plot(fig, image_filename="plotly_train_val_acc.html", auto_open=show)
    # print("Plot saved as plotly_train_val_acc.html")


def plot_train_val_loss(losses, show=True):
    """Plot training vs. testing loss over all epochs."""
    x = list(losses.keys())
    y_train = [i[0] for i in losses.values()]
    y_test = [i[1] for i in losses.values()]

    trace_train = graphs.Scatter(x=x, y=y_train, name="Training", mode="lines+markers",
                                 line=dict(width=4),
                                 marker=dict(symbol="circle",
                                             size=10))
    trace_test = graphs.Scatter(x=x, y=y_test, name="Validation", mode="lines+markers",
                                line=dict(width=4),
                                marker=dict(symbol="circle",
                                            size=10))

    layout = graphs.Layout(title="Training vs. Validation loss",
                           xaxis={"title": "Epoch"},
                           yaxis={"title": "Loss"})

    fig = graphs.Figure(data=[trace_train, trace_test], layout=layout)
    ply.plot(fig, image_filename="plotly_train_val_loss.html", auto_open=show)
    # print("Plot saved as plotly_train_val_loss.html")


def predict(samples, model, show=True):
    """Return class predicted by model for the given samples."""
    predictions = model.predict_generator(generator=samples, use_multiprocessing=False, workers=6)
    if show:
        print("-------")
        for prediction in predictions:
            print(np.argmax(prediction))

    return [np.argmax(prediction) for prediction in predictions]


def create_2DCNN_model(input_shape):
    """Build architecture of the model."""
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=input_shape,
                            activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="selu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation="selu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="selu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="selu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="selu",
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation="selu"))
    model.add(layers.Dense(10, activation="softmax"))

    # Create model
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def main():

    # Data directory
    data_dir = "C:\\Users\\david.haberl\\PycharmProjects\\ProjectThesis2\\Data\\Mnist2D\\"

    # Define hyperparameters
    num_epochs = 5
    batch_size = 32
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    # Get dimensions of one sample
    dims = get_dims(data_dir)

    # Get and map labels to sample IDs
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"), sep=";", header=0)

    labels = dict(zip(labels_df.iloc[:, 0].tolist(), labels_df.iloc[:, 1].tolist()))

    # Create ID-wise training / validation partitioning
    partition = split_data(ids=list(labels.keys()),
                           train_ratio=train_ratio,
                           validation_ratio=validation_ratio,
                           test_ratio=test_ratio)

    # Load data
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

    training_images = DataGenerator(data_dir=data_dir, list_ids=partition["train"], labels=labels,
                                    batch_size=batch_size, dim=dims[0:2], n_channels=1, n_classes=10, shuffle=True,
                                    **da_parameters)

    validation_images = DataGenerator(data_dir=data_dir, list_ids=partition["validation"], labels=labels,
                                      batch_size=batch_size, dim=dims[0:2], n_channels=1, n_classes=10, shuffle=True,
                                      **da_parameters)

    test_images = DataGenerator(data_dir=data_dir, list_ids=partition["test"], labels=labels,
                                batch_size=batch_size, dim=dims[0:2], n_channels=1, n_classes=10, shuffle=True)

    # Create/Compile CNN model
    model = create_2DCNN_model(dims)

    # Train model
    train_summary = model.fit_generator(generator=training_images, validation_data=validation_images,
                                        use_multiprocessing=True, workers=6, epochs=num_epochs)

    print(train_summary.history)

    # Evaluate fitted model using test data
    test_loss, test_acc = model.evaluate_generator(generator=test_images, use_multiprocessing=True, workers=6)
    print("\nTest ACC:", round(test_acc, 3))

# =============================================================================
    # Optional functions

    # Get epochwise performances
    # train_acc = train_summary.history["acc"]
    # val_acc = train_summary.history["val_acc"]

    # train_loss = train_summary.history["loss"]
    # val_loss = train_summary.history["val_loss"]

    # Format and store performances per epoch for plotting
    # accs = {epoch: [round(performance[0], 2), round(performance[1], 2)]
    #         for epoch, performance in enumerate(zip(train_acc, val_acc))}
    # losses = {epoch: [round(performance[0], 2), round(performance[1], 2)]
    #           for epoch, performance in enumerate(zip(train_loss, val_loss))}

    # Plot training and validation performance over epochs
    # plot_train_val_acc(accs)
    # plot_train_val_loss(losses)

    # Plot incorrectly predicted samples TODO: testing required when using Data Generator
    # plot_incorrects(model, test_images, test_labels, num_to_show=5)

    # Prediction of query sample TODO: testing required when using Data Generator
    # predicted = predict(np.array([test_images[0]]), model, show=True)
    # print("Predicted label:", np.argmax(predicted))
    # print("True label:", np.argmax(test_labels[0]))

    # Print model summary including parameters and architecture
    # print(model.summary())

    # Check output of DataGenerator
    # array = training_images.__getitem__(0)

    # Create a grid of 3x3 images
    # for i in range(0, 9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(training_images.__getitem__(0)[0][0:9][i][:, :, 0])
    # plt.show()

    # Plot an image
    # plt.imshow(training_images.__getitem__(0)[0][0][:, :, 0])
    # plt.show()

# =============================================================================


if __name__ == "__main__":
    main()
