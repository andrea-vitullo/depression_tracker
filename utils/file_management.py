import os
import shutil
import pandas as pd

import my_config


def wave_mover():
    """
    Copies audio files from a specified source directory to a target directory.

    This function walks through the file hierarchy rooted at `my_config.DIRECTORY`,
    and for every file that ends with `my_config.FINAL_FORMAT`, it copies
    the file from its current location to the directory specified by `my_config.AUDIO_TEST_DIRECTORY`.

    Args:
        -
    Returns:
        None
    """

    for path, directories, files in os.walk(my_config.DIRECTORY):
        for file in files:
            if file.endswith(my_config.FINAL_FORMAT):
                source_file = os.path.join(path, file)
                target_file = os.path.join(my_config.AUDIO_TEST_DIRECTORY, file)
                shutil.copy(source_file, target_file)


def save_outputs(features, audio, path):
    """Save features to a CSV file."""

    df = pd.DataFrame(features)
    # n = 5  # The number of columns to keep (replace as needed)
    # df = df.iloc[:, :n]

    output_filename = os.path.join(path, f'{audio[0:3]}_features.csv')
    df.to_csv(output_filename, index=False)
    print(f"Saving .csv files: {output_filename}\n")
