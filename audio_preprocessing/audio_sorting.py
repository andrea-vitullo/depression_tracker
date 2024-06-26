import pandas as pd
import os

import my_config
from utils import file_utils


train_df = pd.read_csv(my_config.TRAIN_SPLIT)
dev_df = pd.read_csv(my_config.DEV_SPLIT)
test_df = pd.read_csv(my_config.TEST_SPLIT)

source_dir = my_config.AUDIO_FEATURES

train_dir_0 = my_config.AUDIO_TRAIN_DIR_0
train_dir_1 = my_config.AUDIO_TRAIN_DIR_1

dev_dir_0 = my_config.AUDIO_DEV_DIR_0
dev_dir_1 = my_config.AUDIO_DEV_DIR_1

test_dir_0 = my_config.AUDIO_TEST_DIR_0
test_dir_1 = my_config.AUDIO_TEST_DIR_1

os.makedirs(train_dir_0, exist_ok=True)
os.makedirs(train_dir_1, exist_ok=True)

os.makedirs(dev_dir_0, exist_ok=True)
os.makedirs(dev_dir_1, exist_ok=True)

os.makedirs(test_dir_0, exist_ok=True)
os.makedirs(test_dir_1, exist_ok=True)


def dataframe_iterator(dataframe, dir_0, dir_1):
    """
    Iterate over a pandas DataFrame, extract binary values and gender values, and sort the audio files
    based on these binary values.

    Each row in the DataFrame corresponds to an audio file, identified by 'Participant_ID'.
    Depending on the binary value, the associated audio file is sorted into one of the two directories provided.

    If the DataFrame is the test dataframe, 'PHQ_Binary' is the key for the binary value,
    otherwise 'PHQ8_Binary' is used.

    Args:
        dataframe: A pandas DataFrame with at least 'Participant_ID', 'PHQ_Binary' (or 'PHQ8_Binary')
        and 'Gender' columns.
        dir_0: Path to the directory where audio files with binary value 0 should be sorted to.
        dir_1: Path to the directory where audio files with binary value 1 should be sorted to.

    Returns:
        labels_list: A list of the binary values from the DataFrame in their original order.
    """

    labels_list = []

    for index, row in dataframe.iterrows():
        participant_id = str(int(row['Participant_ID']))  # Ensure participant_id is string type
        if dataframe is test_df:
            binary_value = int(row['PHQ_Binary'])
        else:
            binary_value = int(row['PHQ8_Binary'])  # PHQ8 for train and dev -- PHQ for test

        labels_list.append(binary_value)

        gender_value = int(row['Gender'])
        filename = f"{participant_id}_Final.wav"
        source_file_path = os.path.join(source_dir, filename)

        file_utils.file_sorter(source_file_path, dir_0, dir_1, binary_value, gender_value, filename, source_file_path)

    return labels_list


dataframe_iterator(train_df, train_dir_0, train_dir_1)
dataframe_iterator(dev_df, dev_dir_0, dev_dir_1)
dataframe_iterator(test_df, test_dir_0, test_dir_1)
