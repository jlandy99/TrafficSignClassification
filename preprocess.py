import os
import numpy as np
import pandas as pd
from PIL import Image

from plot import plot_class_dist_and_stats
from config import NUM_TRAIN_IMAGES, NUM_TEST_IMAGES, NUM_TOTAL_IMAGES, IMAGE_DIM, N_CLASS


def preprocess():
    processed_path = 'Data/ProcessedData'
    X_processed_path = os.path.join(processed_path, 'X.npy')
    Y_processed_path = os.path.join(processed_path, 'Y.npy')

    X_processed_path_exists = os.path.exists(X_processed_path)
    Y_processed_path_exists = os.path.exists(Y_processed_path)

    # this will skip processing if our training data already has been processed
    # because this will be a time consuming process and obviously do not want to
    # do this if we do not have to and has already been done
    X_processed = None
    Y_processed = None

    if X_processed_path_exists and Y_processed_path_exists:

      print("X_processed and Y_processed already exist, loading from files")
      X_processed = np.load(X_processed_path)
      Y_processed = np.load(Y_processed_path)

      assert X_processed.shape == (NUM_TOTAL_IMAGES, 3, IMAGE_DIM, IMAGE_DIM)
      assert Y_processed.shape == (NUM_TOTAL_IMAGES,)

      print("X_processed, Y_processed loaded successfully")

    else:

      print("X_processed or Y_processed does not exist, processing raw data")
      print("processing training portion")

      X_processed = np.empty((NUM_TOTAL_IMAGES, 3, IMAGE_DIM, IMAGE_DIM))
      Y_processed = np.empty(NUM_TOTAL_IMAGES)

      # Iterate through each sub folder and access each sub folder's CSV file
      # which contains the image filenames in that subfolder as well as each
      # image's correct classification
      raw_train_dir = 'Data/RawTraining'

      index = 0

      for subfolder in os.listdir(raw_train_dir):

        print('processing', subfolder)

        csv_filename = os.path.join(os.path.join(raw_train_dir, subfolder), 'GT-' + subfolder + '.csv')
        df = pd.read_csv(csv_filename, delimiter=';')

        for _ , row in df.iterrows():

          # open image up/downsample to standard size and insert into X_train
          # record label in Y_train
          img_path = os.path.join(os.path.join(raw_train_dir, subfolder), row['Filename'])
          img = Image.open(img_path).resize((IMAGE_DIM, IMAGE_DIM))

          # RGB channel needs to be second to be compatible with torch net
          X_processed[index] = np.moveaxis(np.asarray(img), -1, 0)
          Y_processed[index] = row['ClassId']
          
          index += 1


      assert index == NUM_TRAIN_IMAGES

      # np.save(X_train_path, X_train)
      # np.save(Y_train_path, Y_train)

      print("X training and Y training processed successfully")
      print("processing testing portion")

      # For testing all images are in one subfolder unlike training
      # in which they were a bunch of different subfolders
      raw_test_dir = 'Data/RawTesting'

      csv_filename = os.path.join(raw_test_dir, 'GT-final_test.csv')
      df = pd.read_csv(csv_filename, delimiter=';')


      for _ , row in df.iterrows():

        # open image up/downsample to standard size and insert into X_test, Y_test
        img_path = os.path.join(raw_test_dir, row['Filename'])
        img = Image.open(img_path).resize((IMAGE_DIM, IMAGE_DIM))

        # RGB channel needs to be second to be compatible with torch net
        X_processed[index] = np.moveaxis(np.asarray(img), -1, 0)
        Y_processed[index] = row['ClassId']
        
        index += 1

        # print some updates to console
        if index % 500 == 0:
          print("processed", index, "testing images")


      assert index == NUM_TOTAL_IMAGES

      np.save(X_processed_path, X_processed)
      np.save(Y_processed_path, Y_processed)

      print("Successfully processed all raw data and saved to numpy arrays for future loading")
      
      """Now we will analyze the distribution of the classes in the training and testing data. We want to make sure the dataset is balanced and not heavily biased towards a specific class so we will balance the dataset here"""

    print('\nCombined Dataset Statistics -----------------------------------------')
    plot_class_dist_and_stats(Y_processed, N_CLASS, 'plots/original_distribution.png')
    
    return X_processed, Y_processed
    

def rebalance(X_processed, Y_processed):
    """Here we can see out dataset is incredibly unbalanced. Completely balancing the dataset would cause us to lose a lot of data so we wll cap the number of training samples for a class at the median."""

    class_max_samples = int(np.median(np.bincount(Y_processed.astype(np.uint8))))

    Y_processed_copy = Y_processed.copy().astype(np.uint8)

    indices_to_keep = []
    for i in range(N_CLASS):

      samples = np.argwhere(Y_processed_copy == i).flatten()
      if samples.shape[0] > class_max_samples:
        samples = samples[:class_max_samples]
      
      indices_to_keep.extend(list(samples))


    # sort the new indices
    indices_to_keep.sort()

    # These variables will be used for training
    new_X_processed = X_processed[indices_to_keep]
    new_Y_processed = Y_processed[indices_to_keep]
    NEW_NUM_TOTAL_IMAGES = new_Y_processed.shape[0]

    print("\nMore Balanced Combined Dataset Statistics")
    plot_class_dist_and_stats(new_Y_processed, N_CLASS, 'plots/balanced_distribution.png')
    
    return new_X_processed, new_Y_processed, NEW_NUM_TOTAL_IMAGES

