import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from sklearn.utils import shuffle

from config import TRAIN_RATIO, VAL_RATIO, NUM_BATCHES


class TrafficSignDataset(Dataset):

  def __init__(self, X, y, data_range):

      # skLearn.utils.shuffle "Shuffles arrays or sparse matrices in a consistent way"
      # we want to shuffle the arrays so they are not always the exact same when training
      # make sures numpy arrays are uint8 type so arent excessvely big for pixel data
      # TODO WE MIGHT NOT NEED THIS CUZ DATALOADER MIGHT SHUFFLE ANYWAYS

      self.X = torch.from_numpy(X[data_range[0]:data_range[1]]).float()
      self.y = torch.from_numpy(y[data_range[0]:data_range[1]]).long()

  def __len__(self):
      return len(self.X)

  def __getitem__(self, index):
      return self.X[index], self.y[index]
      
      
def dataLoader(X_processed, Y_processed, TOTAL_IMAGES):

    #  shuffle because classes are all next to each other based on how data was loaded
    X_processed, Y_processed = shuffle(X_processed.astype(np.uint8), Y_processed.astype(np.uint8))

    TRAIN_SPLIT_SIZE = int(TOTAL_IMAGES * TRAIN_RATIO)
    VAL_SPLIT_SIZE = int(TOTAL_IMAGES * VAL_RATIO)

    train_range = (0, TRAIN_SPLIT_SIZE)
    val_range = (TRAIN_SPLIT_SIZE, TRAIN_SPLIT_SIZE + VAL_SPLIT_SIZE)
    test_range = (TRAIN_SPLIT_SIZE + VAL_SPLIT_SIZE, TOTAL_IMAGES)

    train_data = TrafficSignDataset(X_processed, Y_processed, train_range)
    val_data = TrafficSignDataset(X_processed, Y_processed, val_range)
    test_data = TrafficSignDataset(X_processed, Y_processed, test_range)

    train_loader = DataLoader(train_data, batch_size=int(TRAIN_SPLIT_SIZE / (NUM_BATCHES - 1)), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=int(VAL_SPLIT_SIZE / (NUM_BATCHES - 1)), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)
    
    return train_loader, val_loader, test_loader
