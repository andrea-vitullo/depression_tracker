import glob
import h5py
import numpy as np
import os
import librosa
from sklearn.utils import class_weight

import my_config


def calculate_class_weights(all_labels):
    # This will compute the weights for each class to be applied during training
    class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=all_labels)
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict


def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % my_config.EPOCHS_DROP == 0:
        return lr * my_config.DECAY_FACTOR
    return lr


def load_files_labels(base_dir):
    """
    Loads audio files and their corresponding labels from a given directory.

    This function scans through subdirectories within a base directory according to
    predefined patterns, and, for each .wav file, it adds the file path and its corresponding
    label to the lists. The labels are mapped from the patterns detected in the file path such
    as 'non_depressed/male', 'non_depressed/female', 'depressed/male', 'depressed/female'.

    Args:
        base_dir (str): The base directory containing the audio files.

    Returns:
        files (list): List of audio file paths.
        labels (list): Corresponding list of labels for the audio files.
    """

    files = []
    labels = []

    # Define the patterns for directory scanning
    patterns = ['non_depressed/male', 'non_depressed/female', 'depressed/male', 'depressed/female']
    label_mapping = {
        'non_depressed/male': 0,
        'non_depressed/female': 1,
        'depressed/male': 2,
        'depressed/female': 3,
    }

    print(f"Loading files from {base_dir}")

    # Walk through the base directory
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern, '*.wav')
        for file_path in glob.glob(full_pattern):
            files.append(file_path)
            labels.append(label_mapping[pattern])

            print(f"Loaded {len(files)} files with labels: {set(labels)}")

    print(f"Total loaded files: {len(files)}")

    return files, labels


def load_features(file_path):
    """
    Loads audio data and labels from an HDF5 file.

    This function opens an HDF5 file and reads each key's 'audio' data and its 'label' attribute.
    Assumes that all audio samples are of the same shape and that labels
    are scalar values.

    Args:
        file_path (str): The filename of the HDF5 file.

    Returns:
        audio_data (np.ndarray): Numpy array of the audio data.
        labels (np.ndarray): Corresponding numpy array of labels.
    """

    with h5py.File(file_path, 'r') as h5f:

        num_samples = len(h5f.keys())
        sample_shape = h5f[list(h5f.keys())[0]]['audio'].shape

        # Pre-allocate arrays
        audio_data = np.zeros((num_samples,) + sample_shape)
        labels = np.zeros((num_samples, 1))

        for i, key in enumerate(h5f.keys()):
            audio_data[i] = h5f[key]['audio'][:]
            labels[i] = h5f[key].attrs['label']

        return audio_data, labels


def compute_global_stats_from_test_data(audio_files_directory):
    """
    Computes the global mean and standard deviation from all test files.

    Args:
        audio_files_directory (str): Path to the directory containing test files.

    Returns:
        tuple: A tuple containing the global mean and standard deviation of the Mel spectrograms.
    """

    all_audio_files = []

    audio_files = [f for f in os.listdir(audio_files_directory) if f.endswith('.wav')]

    for audio_file in audio_files:
        audio_file_path = os.path.join(audio_files_directory, audio_file)
        audio, sr = librosa.load(audio_file_path, sr=None)  # Load audio at its native sampling rate

        all_audio_files.append(audio.flatten())

    # Concatenate all audio files into a single 1D array
    all_audio = np.concatenate(all_audio_files)

    # Calculate global mean and std
    global_mean = np.mean(all_audio)
    global_std = np.std(all_audio)

    return global_mean, global_std
