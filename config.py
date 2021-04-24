import torch.nn as nn

### Contains constants for use in rest of files

# Image dimensions (square AxA)
IMAGE_DIM = 64

# Ratio splits for training, validation, and testing
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Model Hyperparamters
CRITERION = nn.CrossEntropyLoss()
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 1
DROPOUT_PROB = 0.4
NUM_BATCHES = 80

### DO NOT CHANGE CONSTANTS BELOW THIS LINE

# Number of *initial* training/testing images and classes
NUM_TRAIN_IMAGES = 39209
NUM_TEST_IMAGES = 12630
NUM_TOTAL_IMAGES = NUM_TRAIN_IMAGES + NUM_TEST_IMAGES
N_CLASS = 43
