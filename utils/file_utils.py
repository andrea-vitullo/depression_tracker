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
                target_file = os.path.join(my_config.AUDIO_FEATURES, file)
                shutil.copy(source_file, target_file)


def save_outputs(features, audio, path):
    """
    Save given features to a CSV file.

    This function takes a list of feature arrays, converts it to a DataFrame,
    and then writes that DataFrame to a CSV file. The output CSV file is saved
    to the specified path with a name based on the input audio file name.

    It prints a message to inform where the .csv file has been saved.

    Args:
        features (array-like): A list of audio feature arrays.
        audio (str): Name of the audio file. The first three characters of this
            name are used as the base for the output filename.
        path (str): Directory path where the output CSV file will be saved.

    Returns:
        None
    """

    df = pd.DataFrame(features)
    # n = 5  # The number of columns to keep (replace as needed)
    # df = df.iloc[:, :n]

    output_filename = os.path.join(path, f'{audio[0:3]}_features.csv')
    df.to_csv(output_filename, index=False)
    print(f"Saving .csv files: {output_filename}\n")


def file_sorter(file_path, dir_0, dir_1, binary_value, gender_value, filename, source_file_path):
    """
    Sorts and moves a file to a specific directory based on a binary value and gender.

    Args:
        file_path (str): The path of the file to be moved.
        dir_0 (str): The destination directory for files with a binary value of 0.
        dir_1 (str): The destination directory for files with a binary value of 1.
        binary_value (int): Binary flag (0 or 1) to determine the destination directory for depression status.
        gender_value (int): Binary flag (0 or 1) to determine the subdirectory for gender within the depression status.
        filename (str): The name of the file to be moved.
        source_file_path (str): Path of the file in the source directory.

    Returns:
        None
    """

    if os.path.exists(source_file_path):
        # Determine main destination directory based on binary_value
        main_dest_dir = dir_1 if binary_value == 1 else dir_0

        # Determine subdirectory based on gender_value
        gender_dir = "male" if gender_value == 1 else "female"
        dest_dir = os.path.join(main_dest_dir, gender_dir)

        # Create the gender subdirectory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Create destination file path
        dest_file_path = os.path.join(dest_dir, filename)

        # Move the file to the appropriate destination directory
        shutil.move(file_path, dest_file_path)
        print(f'Moved file: {filename} to {dest_dir}\n')
    else:
        print(f'File not found: {source_file_path}\n')

wave_mover()