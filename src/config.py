import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model configurations
IMG_SIZE = 80  # Image size (square)
IMG_CHANNELS = 1  # Grayscale=1, RGB=3
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Model architecture
CONV_FILTERS = 64
CONV_KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_UNITS = 64

# Training configurations
MODEL_CHECKPOINT = os.path.join(MODELS_DIR, 'best_model.h5')
SUBMISSION_FILE = os.path.join(SUBMISSIONS_DIR, 'predictions.csv')
