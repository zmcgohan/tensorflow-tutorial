# Tensorflow Tutorial

## About

Tensorflow is an open source library for large-scale machine learning and numerical computation projects in Python.  Tensorflow handles the details of neural networking allowing a developer to focus more on the usage of a neural net as opposed to the implementation of the neural net.  This is useful for image recognition, natural language processing, digit classification, along with many other applications.

Keras is a high-level API meant to run on top of TensorFlow, CNTK, or Theano.  It is made for user-friendliness as it seeks to minimize the number of user interactions while presenting actionable feedback upon error.  Keras runs on the CPU and the GPU along with supporting convolutional networks, recurrent networks or a combination of the two.

Tensorflow 2.0 introduces eager execution which gives first-class status to the built-in Keras modules which are used in this tutorial. The eager execution model allows developers to iterate more quickly through different model architectures than previous, more verbose and lower-level Tensorflow code allowed for while still providing just as much flexibility in model function.

## Installation

To install  

## Training MNIST

MNIST is a dataset consisting of tens of thousands of black-and-white handwritten digits. For years it has been one of the standard datasets researchers use to verify that their learning models function properly. 

In order to learn Tensorflow, we will describe how to use its eager execution model (Keras) to load, preprocess and train a model to classify handwritten digits using the MNIST dataset.

### Importing the modules

First, create a new Python file and import the necessary Tensorflow libraries, the layers we will need for our model and NumPy for preprocessing data. Given the MNIST dataset's popularity, Tensorflow's Keras module provides a built-in way to load it quickly which we will also import.

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np
```

### Preparing the data

Now we will load and prepare the data for training. 

```python
(x_train, y_train_i), (x_test, y_test_i) = mnist.load_data()
```

`x_train` and `x_test` are the grayscale images of handwritten digits, while `y_train_i` and `y_test_i` are their labels (0-9).

For the grayscale images to be used with the convolutional layers of our network, they will need an extra dimension as it expects three dimensions per image.

```python
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
```

And for the labels to be used with the model, they must be converted to one-hot vectors. These are vectors where all values are zero except for the index of the label. E.g., for the digit 5 a one-hot vector would be `[0,0,0,0,0,1,0,0,0,0]`.

```python
y_train, y_test = [np.zeros((len(y), 10)) for y in [y_train_i, y_test_i]]
y_train[np.arange(len(y_train)), y_train_i] = 1.0
y_test[np.arange(len(y_test)), y_test_i] = 1.0
```

### Creating the model

We will now create the convolutional neural network model to train on the MNIST samples. Convolutional neural networks are particularly useful for learning image data.

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.binary_crossentropy,
                optimizer='rmsprop',
                metrics=['accuracy'])
```

### Training 

We can now train our model. With the code below, the model will be trained for 1 epoch with a batch size of 64 samples. Training may take a while depending on your system.

```python
model.fit(x_train, y_train,
              epochs=1,
              batch_size=64,
              verbose=1,
              validation_data=(x_test, y_test)
              )
```

### Evaluating the model

With the model now trained, we can check just how well it performs. The code below will predict the digit of each image in the test set and calculate and print the percent it got correct. From our training, one epoch should reach 97%+ accuracy on the test set.

```python
predictions = model.predict(x_test)
num_correct = len([i for i, pred in enumerate(predictions) if np.argmax(pred) == np.argmax(y_test[i])])
percent_correct = num_correct / float(len(predictions))
print('Test set prediction accuracy: {}%'.format(percent_correct * 100))
```