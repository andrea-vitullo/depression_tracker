import pandas as pd
import shutil
import os

import my_config

# Load the updated csv file
train_df = pd.read_csv(my_config.TRAIN_SPLIT)
dev_df = pd.read_csv(my_config.DEV_SPLIT)
test_df = pd.read_csv(my_config.TEST_SPLIT)

# specify the current directory where your png files are
source_dir = my_config.MFCC_FEATURES

# specify the directory where you want to move the png files
train_dir_0 = my_config.TRAIN_DIR_0
train_dir_1 = my_config.TRAIN_DIR_1

dev_dir_0 = my_config.DEV_DIR_0
dev_dir_1 = my_config.DEV_DIR_1

test_dir_0 = my_config.TEST_DIR_0
test_dir_1 = my_config.TEST_DIR_1

os.makedirs(train_dir_0, exist_ok=True)
os.makedirs(train_dir_1, exist_ok=True)

os.makedirs(dev_dir_0, exist_ok=True)
os.makedirs(dev_dir_1, exist_ok=True)

os.makedirs(test_dir_0, exist_ok=True)
os.makedirs(test_dir_1, exist_ok=True)


def file_sorter(file_path, dir_0, dir_1, binary_value, filename, source_file_path):
    if os.path.exists(source_file_path):

        # Determine destination directory based on binary_value
        dest_dir = dir_1 if binary_value == 1 else dir_0

        # Create destination file path
        dest_file_path = os.path.join(dest_dir, filename)

        # Move the file to the appropriate destination directory
        shutil.move(file_path, dest_file_path)
        print(f'Moved file: {filename} to {dest_dir}\n')
    else:
        print(f'File not found: {source_file_path}\n')


def dataframe_iterator(dataframe, dir_0, dir_1):
    for index, row in dataframe.iterrows():
        participant_id = str(int(row['Participant_ID']))  # Ensure participant_id is string type
        binary_value = int(row['PHQ8_Binary'])  # PHQ8 for train and dev -- PHQ for test
        filename = f"{participant_id}_features.csv" or f"{participant_id}_augmented_features.csv"
        source_file_path = os.path.join(source_dir, filename)

        file_sorter(source_file_path, dir_0, dir_1, binary_value, filename, source_file_path)


dataframe_iterator(train_df, train_dir_0, train_dir_1)
dataframe_iterator(dev_df, dev_dir_0, dev_dir_1)
dataframe_iterator(test_df, test_dir_0, test_dir_1)
