import pandas as pd
import os

import my_config
from utils import file_utils

train_df = pd.read_csv(my_config.TRAIN_SPLIT)
dev_df = pd.read_csv(my_config.DEV_SPLIT)
test_df = pd.read_csv(my_config.TEST_SPLIT)

source_dir = my_config.FORMANT_FEATURES

train_dir_0 = my_config.FORM_TRAIN_DIR_0
train_dir_1 = my_config.FORM_TRAIN_DIR_1

dev_dir_0 = my_config.FORM_DEV_DIR_0
dev_dir_1 = my_config.FORM_DEV_DIR_1

test_dir_0 = my_config.FORM_TEST_DIR_0
test_dir_1 = my_config.FORM_TEST_DIR_1

os.makedirs(train_dir_0, exist_ok=True)
os.makedirs(train_dir_1, exist_ok=True)

os.makedirs(dev_dir_0, exist_ok=True)
os.makedirs(dev_dir_1, exist_ok=True)

os.makedirs(test_dir_0, exist_ok=True)
os.makedirs(test_dir_1, exist_ok=True)


def dataframe_iterator(dataframe, dir_0, dir_1):
    for index, row in dataframe.iterrows():
        participant_id = str(int(row['Participant_ID']))  # Ensure participant_id is string type
        if dataframe is test_df:
            binary_value = int(row['PHQ_Binary'])
        else:
            binary_value = int(row['PHQ8_Binary'])  # PHQ8 for train and dev -- PHQ for test
        filename = f"{participant_id}_features.csv" or f"{participant_id}_augmented_features.csv"
        source_file_path = os.path.join(source_dir, filename)

        file_utils.file_sorter(source_file_path, dir_0, dir_1, binary_value, filename, source_file_path)


dataframe_iterator(train_df, train_dir_0, train_dir_1)
dataframe_iterator(dev_df, dev_dir_0, dev_dir_1)
dataframe_iterator(test_df, test_dir_0, test_dir_1)
