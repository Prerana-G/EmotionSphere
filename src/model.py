from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        # First convolutional layer
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional layer
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten the output for the fully connected layer
        Flatten(),
        
        # Fully connected layer
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer: 7 neurons (one for each emotion), with softmax activation
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with Adam optimizer and categorical crossentropy loss for multi-class classification
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
if __name__ == '__main__':
    model = create_model(input_shape=(48, 48, 1), num_classes=7)
    model.summary()  # Print model summary to check architecture details
