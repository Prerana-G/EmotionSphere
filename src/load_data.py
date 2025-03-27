import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define directories for training and testing
train_dir = '../fer2013/train'
test_dir = '../fer2013/test'

# Initialize ImageDataGenerator for loading images from directories
def load_data(train_dir, test_dir):
    # Initialize generators as None
    train_generator = None
    validation_generator = None

    if train_dir:  # Only initialize train_datagen if train_dir is provided
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            subset='training',
        )

        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            subset='validation',
        )

    if test_dir:  # Only initialize test_datagen if test_dir is provided
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
        )
    else:
        test_generator = None

    # Return the generators
    return train_generator, validation_generator, test_generator

# Display class labels (after data loading)
def display_class_labels(generator):
    print(f"Class labels: {generator.class_indices}")

# Example usage:
if __name__ == '__main__':
    # Call the function to load data and return generators
    train_generator, validation_generator, test_generator = load_data(train_dir, test_dir)

    # Display class labels
    display_class_labels(train_generator)
