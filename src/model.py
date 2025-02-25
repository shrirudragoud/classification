from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
from config import *

def create_model():
    """Create and compile the CNN model."""
    model = Sequential([
        # Input layer
        Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
        
        # First Convolutional Layer
        Conv2D(
            filters=CONV_FILTERS,
            kernel_size=CONV_KERNEL_SIZE,
            activation='relu'
        ),
        MaxPooling2D(pool_size=POOL_SIZE),
        
        # Second Convolutional Layer
        Conv2D(
            filters=CONV_FILTERS,
            kernel_size=CONV_KERNEL_SIZE,
            activation='relu'
        ),
        MaxPooling2D(pool_size=POOL_SIZE),
        
        # Flatten layer to connect to dense layers
        Flatten(),
        
        # Dense hidden layer
        Dense(units=DENSE_UNITS, activation='relu'),
        
        # Output layer
        Dense(units=1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, filepath=MODEL_CHECKPOINT):
    """Save the model to disk."""
    try:
        model.save(filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model_from_checkpoint(filepath=MODEL_CHECKPOINT):
    """Load a saved model from disk."""
    from tensorflow.keras.models import load_model
    try:
        model = load_model(filepath)
        print(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    model.summary()