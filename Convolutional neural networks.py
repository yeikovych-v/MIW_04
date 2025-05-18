import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import kagglehub

ASL_PATH = kagglehub.dataset_download("grassknoted/asl-alphabet")

print("Path to ASL dataset files:", ASL_PATH)


def display_images(train_images):
    fig = plt.figure(figsize=(13, 13))
    fig.tight_layout(pad=0.8, h_pad=2)
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis('off')
        ax.imshow(train_images[i], cmap=plt.cm.binary)
    plt.show()

def train():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

if __name__ == '__main__':
    #Load dataset
    fashion = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion.load_data()
    print(f"Shape of input data: {train_images.shape}")

    model = train()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.show()

    accuracy_level = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Accuracy loss: {accuracy_level[0]}")
    print(f"Accuracy level: {accuracy_level[1]}")
