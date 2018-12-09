from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np

# Load the data
(x_train, y_train_i), (x_test, y_test_i) = mnist.load_data()

# Reshape the inputs for the convolution layers
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Turn labels into one-hots
y_train, y_test = [np.zeros((len(y), 10)) for y in [y_train_i, y_test_i]]
y_train[np.arange(len(y_train)), y_train_i] = 1.0
y_test[np.arange(len(y_test)), y_test_i] = 1.0

# Create the CNN model
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

# Train!
model.fit(x_train, y_train,
    epochs=1,
    batch_size=64,
    verbose=1,
    validation_data=(x_test, y_test)
    )

# Evaluate accuracy against the test set
predictions = model.predict(x_test)
num_correct = len([i for i, pred in enumerate(predictions) if np.argmax(pred) == np.argmax(y_test[i])])
percent_correct = num_correct / float(len(predictions))
print('Test set prediction accuracy: {}%'.format(percent_correct * 100))