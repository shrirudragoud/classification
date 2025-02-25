import os
import numpy as np
import pandas as pd
from data_loader import load_dataset
from model import load_model_from_checkpoint
from config import *

def predict_images():
    """Make predictions on test images and create submission file."""
    # Load test images
    print("Loading test images...")
    try:
        x_test, _, image_paths = load_dataset(TEST_DIR, is_training=False)
        print(f"Loaded {len(x_test)} test images")
    except Exception as e:
        print(f"Error loading test images: {str(e)}")
        return
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_model_from_checkpoint()
    if model is None:
        print("Could not load model. Please ensure model is trained first.")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
    
    # Create submission dataframe
    print("\nCreating submission file...")
    submission_data = []
    
    for img_path, pred in zip(image_paths, predictions):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        # Convert prediction to label (0 = cat, 1 = dog)
        label = 1 if pred[0] >= 0.5 else 0
        submission_data.append({
            'image_id': img_id,
            'prediction': label
        })
    
    # Create and save submission file
    df = pd.DataFrame(submission_data)
    df.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission file saved to {SUBMISSION_FILE}")
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted cats: {len(predictions[predictions < 0.5])}")
    print(f"Predicted dogs: {len(predictions[predictions >= 0.5])}")

def evaluate_model():
    """Evaluate model on training data to check accuracy."""
    # Load training data
    print("Loading training data for evaluation...")
    try:
        x_train, y_train, _ = load_dataset(TRAIN_DIR)
        print(f"Loaded {len(x_train)} training images")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_model_from_checkpoint()
    if model is None:
        print("Could not load model. Please ensure model is trained first.")
        return
    
    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=1)
    print(f"\nEvaluation Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # First evaluate the model
    evaluate_model()
    
    # Then make predictions on test data
    predict_images()