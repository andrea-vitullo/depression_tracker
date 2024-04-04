import glob
import h5py
import numpy as np
import os
import librosa

from my_config import *
# from data_generator import DataGenerator
# from features_extractors import extract_raw_audio, extract_mfcc, extract_logmel


def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % EPOCHS_DROP == 0:
        return lr * DECAY_FACTOR
    return lr


# def create_datagenerator(extraction_function, train_filepath, dev_filepath, test_filepath, batch_size):
#     # Create the appropriate DataGenerator based on extraction_function
#     if extraction_function == extract_raw_audio:
#         train_generator = DataGenerator(train_filepath, batch_size, audio_shape=RAW_SHAPE)
#         dev_generator = DataGenerator(dev_filepath, batch_size, audio_shape=RAW_SHAPE)
#         test_generator = DataGenerator(test_filepath, batch_size, audio_shape=RAW_SHAPE)
#     elif extraction_function == extract_mfcc:
#         train_generator = DataGenerator(train_filepath, batch_size, audio_shape=MFCC_SHAPE)
#         dev_generator = DataGenerator(dev_filepath, batch_size, audio_shape=MFCC_SHAPE)
#         test_generator = DataGenerator(test_filepath, batch_size, audio_shape=MFCC_SHAPE)
#     elif extraction_function == extract_logmel:
#         train_generator = DataGenerator(train_filepath, batch_size, audio_shape=LOGMEL_SHAPE)
#         dev_generator = DataGenerator(dev_filepath, batch_size, audio_shape=LOGMEL_SHAPE)
#         test_generator = DataGenerator(test_filepath, batch_size, audio_shape=LOGMEL_SHAPE)
#     else:
#         raise ValueError("Invalid extraction_function. Must be one of [extract_raw_audio, extract_mfcc, extract_logmel]")
#
#     return train_generator, dev_generator, test_generator


# def load_files_labels(directories, labels):
#     files = []
#     label_list = []
#
#     for directory, label in zip(directories, labels):
#         for gender in ['male', 'female']:
#             gender_files = glob.glob(f"{directory}/{gender}/*.wav")
#             files.extend(gender_files)
#             label_list.extend([label] * len(gender_files))
#
#     return files, label_list


def load_files_labels(base_dir):

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

    with h5py.File(file_path, 'r') as h5f:
        # Let's assume we know the number of samples beforehand
        num_samples = len(h5f.keys())
        # Let's assume all audio data have the same shape
        sample_shape = h5f[list(h5f.keys())[0]]['audio'].shape

        # Pre-allocate arrays
        audio_data = np.zeros((num_samples,) + sample_shape)
        labels = np.zeros((num_samples, 1))  # assuming labels are scalar values

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

    all_mel_specs = []

    # Assuming audio files are in 'wav' format
    audio_files = [f for f in os.listdir(audio_files_directory) if f.endswith('.wav')]

    for audio_file in audio_files:
        audio_file_path = os.path.join(audio_files_directory, audio_file)
        audio, sr = librosa.load(audio_file_path, sr=None)  # Load audio at its native sampling rate

        # Compute Mel spectrogram without applying dB conversion
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)

        all_mel_specs.append(mel_spec.flatten())

    # Concatenate all mel spectrograms into a single 1D array
    all_mel_specs_flat = np.concatenate(all_mel_specs)

    # Calculate global mean and std
    global_mean = np.mean(all_mel_specs_flat)
    global_std = np.std(all_mel_specs_flat)

    return global_mean, global_std
