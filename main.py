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


def get_dims(data_dir):
    """Get dimensions from a npy file in the given data directory data_dir."""
    npy_files = [file for file in os.listdir(data_dir) if file.endswith(".npy")]
    example_path = os.path.join(data_dir, npy_files[0])
    npy_example = np.load(example_path)

    return npy_example.shape


def load_data(ids, labels, data_dir):
    """Returns 2D image arrays and labels of given IDs."""
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
    """Plot training vs. testing accuracy over all epochs."""
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
    ply.plot(fig, image_filename="plotly_train_val_acc.html", auto_open=show)
    # print("Plot saved as plotly_train_val_acc.html")


def predict(samples, model, show=True):
    """Return class predicted by model for the given samples."""
    predictions = model.predict(samples)
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
    data_dir = "data/"

    # Define hyperparameters
    num_epochs = 100
    batch_size = 32
    train_ratio = 0.8
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
    train_images, train_labels = load_data(partition["train"], labels, data_dir)
    validation_images, validation_labels = load_data(partition["validation"], labels, data_dir)
    test_images, test_labels = load_data(partition["test"], labels, data_dir)

    # Create/Compile CNN model
    model = create_2DCNN_model(dims)

    # Train model
    train_summary = model.fit(x=train_images,
                              y=train_labels,
                              validation_data=(validation_images, validation_labels),
                              batch_size=batch_size,
                              epochs=num_epochs
                              )

    print(train_summary.history)

    # Evaluate fitted model using test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
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

    # Plot incorrectly predicted samples
    # plot_incorrects(model, test_images, test_labels, num_to_show=5)

    # Prediction of query sample
    # predicted = predict(np.array([test_images[0]]), model, show=True)
    # print("Predicted label:", np.argmax(predicted))
    # print("True label:", np.argmax(test_labels[0]))

    # Plot an image
    # plt.imshow(test_images[0][:, :, 0])
    # plt.show()

    # Print model summary including parameters and architecture
    # print(model.summary())

# =============================================================================


if __name__ == "__main__":
    main()
