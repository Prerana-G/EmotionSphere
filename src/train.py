import os
import numpy as np
import tensorflow as tf
from load_data import load_data
from model import create_model  
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define directories for training and testing
train_dir = '../fer2013/train'
test_dir = '../fer2013/test'

# Load the data generators (train, validation, test)
train_generator, validation_generator, test_generator = load_data(train_dir, test_dir)

# Create the model
model = create_model(input_shape=(48, 48, 1), num_classes=7)

# Define callbacks to save the best model and stop early if validation accuracy does not improve
callbacks = [
    ModelCheckpoint('fer_cnn_best_model.keras', save_best_only=True, save_weights_only=False, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # You can adjust this depending on when the model starts to overfit
    verbose=2,  # Shows progress bar during training
    callbacks=callbacks  # Add the callbacks to save the best model and handle early stopping
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the final model
model.save('fer_cnn_final_model.h5')


#Plot accuracy and loss curves
import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

