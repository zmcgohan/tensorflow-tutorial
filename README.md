# Tensorflow Tutorial

## About

Tensorflow is an open source library for large-scale machine learning and numerical computation projects in python.  Tensorflow handles the details of neural networking allowing a developer to focus more on the usage of a neural net as opposed to the implementation of the neural net.  This is useful for image recognition, natural language processing, digit classification, along with many other applications.

Keras is a high level API ment to run on top of TensorFlow, CNTK, or Theano.  It is made for user friendliness as it seeks to minimize the number of user interactions while presenting actionable feedback upon error.  Keras runs on the CPU and the GPU along with supporting convolutional networks, recurrent networks or a combination of the two.

## Installation

To install  

## Training MNIST

### Loading the data

### Creating the model

```python
from keras import Sequential

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

### Training 