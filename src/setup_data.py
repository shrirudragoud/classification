import os
import shutil
import kagglehub
from config import *

def setup_dataset():
    """Download and organize the dataset into the project structure."""
    print("Starting dataset setup...")
    
    # Download dataset
    print("\nDownloading dataset...")
    try:
        path = kagglehub.dataset_download("tongpython/cat-and-dog")
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False
    
    # Create required directories if they don't exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, 'cats'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, 'dogs'), exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Organize files into appropriate directories
    try:
        downloaded_files = []
        # Walk through the downloaded directory
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    downloaded_files.append(os.path.join(root, file))
        
        print(f"\nFound {len(downloaded_files)} image files")
        
        # Sort files into train and test sets
        n_files = len(downloaded_files)
        n_train = int(n_files * 0.8)  # Use 80% for training
        
        train_files = downloaded_files[:n_train]
        test_files = downloaded_files[n_train:]
        
        # Process training files
        print("\nOrganizing training files...")
        for file in train_files:
            filename = os.path.basename(file)
            if 'cat' in filename.lower():
                dest = os.path.join(TRAIN_DIR, 'cats', filename)
            else:
                dest = os.path.join(TRAIN_DIR, 'dogs', filename)
            shutil.copy2(file, dest)
        
        # Process test files
        print("\nOrganizing test files...")
        for file in test_files:
            filename = os.path.basename(file)
            dest = os.path.join(TEST_DIR, filename)
            shutil.copy2(file, dest)
        
        print("\nDataset organization complete!")
        print(f"Training images: {len(train_files)}")
        print(f"Test images: {len(test_files)}")
        return True
        
    except Exception as e:
        print(f"\nError organizing dataset: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_dataset():
        print("\nDataset setup completed successfully!")
    else:
        print("\nDataset setup failed. Please check the error messages above.")