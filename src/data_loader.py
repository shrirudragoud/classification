import os
import cv2
import numpy as np
import kagglehub
from config import *
from sklearn.utils import shuffle

def download_dataset():
    """Download the dataset using kagglehub."""
    try:
        print("Downloading dataset...")
        path = kagglehub.dataset_download("tongpython/cat-and-dog")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale if needed
    if IMG_CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Add channel dimension if grayscale
    if IMG_CHANNELS == 1:
        img = np.expand_dims(img, axis=-1)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    return img

def load_dataset(data_dir, is_training=True):
    """Load and preprocess all images from the directory."""
    images = []
    labels = []
    image_paths = []
    
    if is_training:
        # For training data with subdirectories (cats/dogs)
        for class_name in ['cats', 'dogs']:
            class_dir = os.path.join(data_dir, class_name)
            label = 0 if class_name == 'cats' else 1
            
            if not os.path.exists(class_dir):
                raise ValueError(f"Directory not found: {class_dir}")
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = load_and_preprocess_image(img_path)
                        images.append(img)
                        labels.append(label)
                        image_paths.append(img_path)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
    else:
        # For test data (no subdirectories)
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory not found: {data_dir}")
            
        for img_name in os.listdir(data_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(data_dir, img_name)
                try:
                    img = load_and_preprocess_image(img_path)
                    images.append(img)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    # Convert to numpy arrays
    images = np.array(images)
    if is_training:
        labels = np.array(labels)
        # Shuffle the data
        images, labels, image_paths = shuffle(images, labels, image_paths, random_state=42)
        return images, labels, image_paths
    
    return images, None, image_paths

def create_data_generators(x_train, y_train):
    """Create training and validation data generators."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Calculate split indices
    split_idx = int(len(x_train) * (1 - VALIDATION_SPLIT))
    
    # Split the data
    x_train_split = x_train[:split_idx]
    y_train_split = y_train[:split_idx]
    x_val_split = x_train[split_idx:]
    y_val_split = y_train[split_idx:]
    
    # Create data generator with basic augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Create validation generator (no augmentation)
    val_datagen = ImageDataGenerator()
    
    # Create training generator
    train_generator = train_datagen.flow(
        x_train_split,
        y_train_split,
        batch_size=BATCH_SIZE
    )
    
    # Create validation generator
    val_generator = val_datagen.flow(
        x_val_split,
        y_val_split,
        batch_size=BATCH_SIZE
    )
    
    return train_generator, val_generator

if __name__ == "__main__":
    # Test dataset download and loading
    download_path = download_dataset()
    if download_path:
        print("Testing data loading...")
        try:
            x_train, y_train, train_paths = load_dataset(TRAIN_DIR)
            print(f"Loaded {len(x_train)} training images")
            print(f"Data shape: {x_train.shape}")
            print(f"Labels shape: {y_train.shape}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")