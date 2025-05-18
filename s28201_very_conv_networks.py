import numpy as np
import tensorflow as tf
import keras
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import kagglehub

ASL_PATH = kagglehub.dataset_download("grassknoted/asl-alphabet")
ASL_CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                   'nothing', 'space']
IMG_SIZE = (16, 16)
TRAIN_NEW_MODEL = False

print("Path to ASL dataset files:", ASL_PATH)


def load_asl_dataset(dataset_path, img_size=(64, 64)):
    train_path = os.path.join(dataset_path, 'asl_alphabet_train\\asl_alphabet_train')
    test_path = os.path.join(dataset_path, 'asl_alphabet_test\\asl_alphabet_test')

    if not os.path.isdir(train_path):
        raise ValueError(f"Training directory not found: {train_path}")
    if not os.path.isdir(test_path):
        raise ValueError(f"Testing directory not found: {test_path}")

    train_images = []
    train_labels = []

    # Load training data
    print("Loading training data...")
    for idx, class_name in enumerate(ASL_CLASS_NAMES):
        class_path = os.path.join(train_path, class_name)

        if not os.path.isdir(class_path):
            print(f"Warning: Training directory for class '{class_name}' not found, skipping")
            continue

        print(f"Loading class {class_name} (index {idx})...")

        # Load all images for this class
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"Warning: No images found in {class_path}")
            continue

        print(f"  Found {len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Convert to RGB (OpenCV uses BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to target dimensions
            img = cv2.resize(img, img_size)

            train_images.append(img)
            train_labels.append(idx)

    test_images = []
    test_labels = []

    # Load test data
    print("\nLoading test data...")
    test_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in test_files:
        if img_file.lower() == 'nothing_test.jpg':
            class_name = 'nothing'
        elif img_file.lower() == 'space_test.jpg':
            class_name = 'space'
        else:
            class_name = img_file.split('_')[0]

        if class_name in ASL_CLASS_NAMES:
            class_idx = ASL_CLASS_NAMES.index(class_name)
        else:
            print(f"Warning: Unknown class in test file {img_file}, skipping")
            continue

        img_path = os.path.join(test_path, img_file)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Convert to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target dimensions
        img = cv2.resize(img, img_size)

        test_images.append(img)
        test_labels.append(class_idx)

    train_data = np.array(train_images, dtype='float32')
    train_labels = np.array(train_labels)
    test_data = np.array(test_images, dtype='float32')
    test_labels = np.array(test_labels)

    train_data = train_data / 255.0
    test_data = test_data / 255.0

    train_labels = to_categorical(train_labels, num_classes=len(ASL_CLASS_NAMES))
    test_labels = to_categorical(test_labels, num_classes=len(ASL_CLASS_NAMES))

    print(f"\nDataset loaded:")
    print(f"Training: {len(train_data)} images, shape: {train_data[0].shape}")
    print(f"Testing: {len(test_data)} images, shape: {test_data[0].shape}")

    return (train_data, train_labels), (test_data, test_labels)


def create_asl_model(input_shape, num_classes=28):
    model = Sequential()

    # First convolutional block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional block
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model


def display_asl_images(images, predictions=None, labels=None, num_images=25):
    fig = plt.figure(figsize=(15, 15))
    fig.tight_layout(pad=0.8, h_pad=4, w_pad=1)

    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(5, 5, i + 1)

        ax.imshow(images[i])

        if predictions is not None:
            predicted_label = np.argmax(predictions[i])
            predicted_letter = ASL_CLASS_NAMES[predicted_label]

            if labels is not None:
                true_label = np.argmax(labels[i])
                true_letter = ASL_CLASS_NAMES[true_label]

                title_color = 'green' if predicted_label == true_label else 'red'
                title = f"Pred: {predicted_letter}\nTrue: {true_letter}"
            else:
                title_color = 'black'
                title = f"Pred: {predicted_letter}"

            ax.set_title(title, color=title_color, fontsize=12)

        ax.axis('off')

    plt.show()


def predict_asl_image(model, image, preprocess=True, img_size=(64, 64)):
    if preprocess:
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        img = cv2.resize(img, img_size)

        img = img / 255.0
    else:
        img = image

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    predicted_letter = ASL_CLASS_NAMES[predicted_class]

    plt.figure(figsize=(6, 6))
    plt.imshow(img[0])
    plt.title(f"Prediction: {predicted_letter}", fontsize=16)
    plt.axis('off')
    plt.show()

    print(f"Predicted letter: {predicted_letter} (Confidence: {prediction[0][predicted_class]:.4f})")
    print("\nTop predictions:")
    top_indices = np.argsort(prediction[0])[-5:][::-1]
    for idx in top_indices:
        print(f"{ASL_CLASS_NAMES[idx]}: {prediction[0][idx]:.4f}")

    return predicted_letter, prediction[0]


def save_asl_model(model, filename="asl_model.h5"):
    model.save(filename)
    print(f"Model saved to {filename}")


def load_asl_model(filename="asl_model.h5"):
    model = load_model(filename)
    print(f"Model loaded from {filename}")
    return model


def train_asl_model(dataset_path, img_size=(64, 64), epochs=10, batch_size=32):
    (train_data, train_labels), (test_data, test_labels) = load_asl_dataset(dataset_path, img_size=img_size)

    # Determine input shape
    input_shape = train_data[0].shape

    model = create_asl_model(input_shape, num_classes=len(ASL_CLASS_NAMES))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(train_data)

    # Train model with data augmentation
    history = model.fit(
        datagen.flow(train_data, train_labels, batch_size=batch_size),
        epochs=epochs,
        validation_data=(test_data, test_labels),
        steps_per_epoch=len(train_data) // batch_size
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    save_asl_model(model)

    return model


if __name__ == '__main__':
    if TRAIN_NEW_MODEL:
        model = train_asl_model(ASL_PATH, img_size=IMG_SIZE, epochs=10, batch_size=32)
    else:
        model = load_asl_model()

    # Predict and visualize training images
    if not TRAIN_NEW_MODEL:
        (_, _), (X_test, y_test) = load_asl_dataset(ASL_PATH, img_size=IMG_SIZE)

        print("\nPredicting test images:")
        predictions = model.predict(X_test[:25])
        display_asl_images(X_test[:25], predictions, y_test[:25])
