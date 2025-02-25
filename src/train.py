import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import load_dataset, create_data_generators
from model import create_model, save_model
from config import *

def plot_training_history(history, save_path=None):
    """Plot training history and optionally save the plot."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    
    plt.close()

def train_model():
    """Train the model on the dataset."""
    # Load and preprocess data
    print("Loading dataset...")
    try:
        x_train, y_train, _ = load_dataset(TRAIN_DIR)
        print(f"Loaded {len(x_train)} images for training")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator = create_data_generators(x_train, y_train)
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    model.summary()
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_CHECKPOINT,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Calculate steps per epoch and validation steps
    train_samples = int(len(x_train) * (1 - VALIDATION_SPLIT))
    val_samples = int(len(x_train) * VALIDATION_SPLIT)
    steps_per_epoch = train_samples // BATCH_SIZE
    validation_steps = val_samples // BATCH_SIZE
    
    # Calculate class weights
    class_labels = y_train
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("\nUsing class weights:", class_weight_dict)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Add class weights
        verbose=1
    )
    
    # Save training plots
    plot_path = os.path.join(MODELS_DIR, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Save final model
    save_model(model)
    
    return model, history

if __name__ == "__main__":
    # Train the model
    model, history = train_model()
    
    if model is not None:
        # Print final metrics
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f"\nFinal validation accuracy: {final_val_accuracy:.4f}")