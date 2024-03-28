import os
import librosa
from librosa.effects import time_stretch, pitch_shift
import numpy as np
import h5py
import random

import my_config
from my_config import SAMPLERATE, MAX_LENGTH, LABELS
from utils import audio_utils, utils
from features_extractors import extract_logmel, compute_global_mel_stats


# EXTRACTION FUNCTION
# [extract_raw_audio, extract_mfcc, extract_logmel] based on extraction type to perform
# import from features_extractors
EXTRACTION_FUNCTION = extract_logmel


# # Define the number of augmented versions to generate for each depressed class sample
# male_non_depressed_augmentations = 0
# female_non_depressed_augmentations = 0
# male_depressed_augmentations = 1
# female_depressed_augmentations = 1


# Define directory and label mappings
AUDIO_TRAIN_DIRS = [my_config.AUDIO_TRAIN_DIR_0,
                    my_config.AUDIO_TRAIN_DIR_0,
                    my_config.AUDIO_TRAIN_DIR_1,
                    my_config.AUDIO_TRAIN_DIR_1]

AUDIO_DEV_DIRS = [my_config.AUDIO_DEV_DIR_0,
                  my_config.AUDIO_DEV_DIR_0,
                  my_config.AUDIO_DEV_DIR_1,
                  my_config.AUDIO_DEV_DIR_1]

AUDIO_TEST_DIRS = [my_config.AUDIO_TEST_DIR_0,
                   my_config.AUDIO_TEST_DIR_0,
                   my_config.AUDIO_TEST_DIR_1,
                   my_config.AUDIO_TEST_DIR_1]


# Load the data
train_files, train_labels = utils.load_files_labels(AUDIO_TRAIN_DIRS, LABELS)
dev_files, dev_labels = utils.load_files_labels(AUDIO_DEV_DIRS, LABELS)
test_files, test_labels = utils.load_files_labels(AUDIO_TEST_DIRS, LABELS)


global_mel_mean, global_mel_std = compute_global_mel_stats(train_files)


def preprocess_and_save_features(file_paths, labels, output_file_path, augment=False,
                                 extraction_func=EXTRACTION_FUNCTION, mean=None, std=None):
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(output_file_path, 'w') as h5f:
        for i, (file_path, label) in enumerate(zip(file_paths, labels)):
            print(f"Processing file: {file_path}")
            audio, sr = librosa.load(file_path, sr=SAMPLERATE)

            if len(audio) > MAX_LENGTH:
                audio = audio[:MAX_LENGTH]
            elif len(audio) < MAX_LENGTH:
                padding = MAX_LENGTH - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')

            audio_features_padded = extraction_func(audio, sr, mean=mean, std=std)

            # Map multi-class labels to binary labels here
            if label in [0, 1]:    # Non-depressed
                binary_label = 0
            elif label in [2, 3]:  # Depressed
                binary_label = 1

            grp = h5f.create_group(str(i))
            grp.create_dataset('audio', data=audio_features_padded, compression='gzip')
            grp.attrs['label'] = binary_label

            # if augment:
            #     # Determine the number of augmentations based on the label
            #     if binary_label == 0:
            #         num_augmentations = male_non_depressed_augmentations + female_non_depressed_augmentations
            #     elif binary_label == 1:
            #         num_augmentations = male_depressed_augmentations + female_depressed_augmentations
            #
            #     for aug_index in range(num_augmentations):
            #         augmentation_type = random.choice(['noise'])
            #         print(f"Augmentation type: {augmentation_type}")
            #
            #         if augmentation_type == 'noise':
            #             augmented_audio = audio_utils.add_noise(audio)
            #         # elif augmentation_type == 'stretch':
            #         #     stretch_rate = random.uniform(0.9, 1.1)
            #         #     augmented_audio = time_stretch(audio, rate=stretch_rate)
            #         # elif augmentation_type == 'shift':
            #         #     n_steps = random.randint(-1, 1)
            #         #     augmented_audio = pitch_shift(audio, sr=sr, n_steps=n_steps)
            #
            #         augmented_features = EXTRACTION_FUNCTION(augmented_audio, sr, mean=mean, std=std)
            #         aug_grp = h5f.create_group(f"{i}_aug_{aug_index}")
            #         aug_grp.create_dataset('audio', data=augmented_features, compression='gzip')
            #         aug_grp.attrs['label'] = binary_label
            #
            #         print(f"Augmented audio label: {binary_label}")


preprocess_and_save_features(
    train_files,
    train_labels,
    './processed_audio_features/train_features.h5',
    augment=False,
    extraction_func=EXTRACTION_FUNCTION,
    mean=global_mel_mean,
    std=global_mel_std
)

preprocess_and_save_features(
    dev_files,
    dev_labels,
    './processed_audio_features/dev_features.h5',
    augment=False,
    extraction_func=EXTRACTION_FUNCTION,
    mean=global_mel_mean,
    std=global_mel_std
)

preprocess_and_save_features(
    test_files,
    test_labels,
    './processed_audio_features/test_features.h5',
    augment=False,
    extraction_func=EXTRACTION_FUNCTION,
    mean=global_mel_mean,
    std=global_mel_std
)
