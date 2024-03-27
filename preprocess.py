import os
import librosa
import numpy as np
import h5py
import random

from my_config import SAMPLERATE, MAX_LENGTH
from utils import audio_utils
from features_extractors import extract_raw_audio, extract_mfcc, extract_logmel


# EXTRACTION FUNCTION
# [extract_raw_audio, extract_mfcc, extract_logmel] based on extraction type to perform
EXTRACTION_FUNCTION = extract_raw_audio


# Define the number of augmented versions to generate for each depressed class sample
male_non_depressed_augmentations = 0
female_non_depressed_augmentations = 1
male_depressed_augmentations = 2
female_depressed_augmentations = 2


def preprocess_and_save_features(file_paths, labels, output_file_path, augment=False, extraction_func=EXTRACTION_FUNCTION):
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

            audio_features_padded = extraction_func(audio, sr)

            # Map multi-class labels to binary labels here
            if label in [0, 1]:    # Non-depressed
                binary_label = 0
            elif label in [2, 3]:  # Depressed
                binary_label = 1

            grp = h5f.create_group(str(i))
            grp.create_dataset('audio', data=audio_features_padded, compression='gzip')
            grp.attrs['label'] = binary_label

            if augment:
                # Determine the number of augmentations based on the label
                if binary_label == 0:
                    num_augmentations = male_non_depressed_augmentations + female_non_depressed_augmentations
                elif binary_label == 1:
                    num_augmentations = male_depressed_augmentations + female_depressed_augmentations

                for aug_index in range(num_augmentations):
                    augmentation_type = random.choice(['noise'])
                    print(f"Augmentation type: {augmentation_type}")

                    if augmentation_type == 'noise':
                        augmented_audio = audio_utils.add_noise(audio)
                    # elif augmentation_type == 'stretch':
                        # stretch_rate = random.uniform(0.9, 1.1)
                        # augmented_audio = time_stretch(audio, rate=stretch_rate)
                    # elif augmentation_type == 'shift':
                    #     n_steps = random.randint(-1, 1)
                    #     augmented_audio = pitch_shift(audio, sr=sr, n_steps=n_steps)

                    augmented_mfcc_features = EXTRACTION_FUNCTION(augmented_audio, sr)
                    aug_grp = h5f.create_group(f"{i}_aug_{aug_index}")
                    aug_grp.create_dataset('audio', data=augmented_mfcc_features, compression='gzip')
                    aug_grp.attrs['label'] = binary_label

                    print(f"Augmented audio label: {binary_label}")
