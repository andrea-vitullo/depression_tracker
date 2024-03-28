import os
import librosa
from librosa.effects import time_stretch, pitch_shift
import numpy as np
import h5py
import random
import re

import my_config
from my_config import SAMPLERATE, MAX_LENGTH, LABELS
from utils import audio_utils, utils
from features_extractors import extract_logmel_segments, compute_global_mel_stats


# EXTRACTION FUNCTION
# [extract_raw_audio, extract_mfcc, extract_logmel] based on extraction type to perform
# import from features_extractors
EXTRACTION_FUNCTION = extract_logmel_segments


# AUDIO_TRAIN_DIRS = [my_config.AUDIO_TRAIN_DIR_0,
#                     my_config.AUDIO_TRAIN_DIR_0,
#                     my_config.AUDIO_TRAIN_DIR_1,
#                     my_config.AUDIO_TRAIN_DIR_1]
#
# AUDIO_DEV_DIRS = [my_config.AUDIO_DEV_DIR_0,
#                   my_config.AUDIO_DEV_DIR_0,
#                   my_config.AUDIO_DEV_DIR_1,
#                   my_config.AUDIO_DEV_DIR_1]
#
# AUDIO_TEST_DIRS = [my_config.AUDIO_TEST_DIR_0,
#                    my_config.AUDIO_TEST_DIR_0,
#                    my_config.AUDIO_TEST_DIR_1,
#                    my_config.AUDIO_TEST_DIR_1]


# Load the data
train_files, train_labels = utils.load_files_labels(my_config.AUDIO_TRAIN_DIR)
dev_files, dev_labels = utils.load_files_labels(my_config.AUDIO_DEV_DIR)
test_files, test_labels = utils.load_files_labels(my_config.AUDIO_TEST_DIR)


global_mel_mean, global_mel_std = compute_global_mel_stats(train_files)


######################################################################################################################


def balance_and_select_speakers(file_paths, labels, speaker_ids, num_speakers_per_category=31):
    """
    Selects an equal number of speakers for each category to ensure balanced representation.

    Args:
    - file_paths (list): List of all file paths.
    - labels (list): Corresponding labels for each file path.
    - speaker_ids (list): Unique identifier for each speaker in the file paths.
    - num_speakers_per_category (int): Number of speakers to select per category.

    Returns:
    - Balanced file paths, labels, and speaker IDs.
    """

    from collections import defaultdict
    category_speakers = defaultdict(list)

    # Map each speaker to their category
    for file_path, label, speaker_id in zip(file_paths, labels, speaker_ids):
        category_speakers[label].append(speaker_id)

    selected_file_paths = []
    selected_labels = []
    selected_speaker_ids = []

    # For each category, randomly select `num_speakers_per_category`
    for label, speakers in category_speakers.items():
        selected_speakers = np.random.choice(speakers, num_speakers_per_category, replace=False)

        # Filter the original lists to include only selected speakers
        for file_path, label, speaker_id in zip(file_paths, labels, speaker_ids):
            if speaker_id in selected_speakers:
                selected_file_paths.append(file_path)
                selected_labels.append(label)
                selected_speaker_ids.append(speaker_id)

    return selected_file_paths, selected_labels, selected_speaker_ids


######################################################################################################################


def preprocess_and_save_features(file_paths, labels, output_file_path, extraction_func, mean=None, std=None,
                                 optimum_segments=89, speakers_per_class=None):

    print(f"Creating directory {os.path.dirname(output_file_path)} if it doesn't exist...")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Map speaker IDs to their audio files and labels
    speaker_data = {}

    print("Mapping speaker IDs to their files and labels...")
    for path, label in zip(file_paths, labels):
        speaker_id = re.findall(r"(\d+)_Final.wav", path)[0]
        if speaker_id not in speaker_data:
            speaker_data[speaker_id] = {'files': [], 'label': label}
            print(f"Speaker ID {speaker_id} added to speaker_data with label {label}.")

        speaker_data[speaker_id]['files'].append(path)

    # Determine the number of speakers to select per class if not specified
    if not speakers_per_class:
        class_counts = {label: 0 for label in set(labels)}
        for speaker in speaker_data.values():
            class_counts[speaker['label']] += 1
        speakers_per_class = min(class_counts.values())
        print(f"Speakers per class determined: {speakers_per_class}")

    # Select a balanced set of speakers for each class
    print("Selecting a balanced set of speakers for each class...")
    balanced_speakers = {label: [] for label in set(labels)}
    for speaker_id, data in speaker_data.items():
        if len(balanced_speakers[data['label']]) < speakers_per_class:
            balanced_speakers[data['label']].append(speaker_id)
            print(f"Speaker ID {speaker_id} with label {data['label']} selected.")

    with h5py.File(output_file_path, 'w') as h5f:
        print(f"Opened {output_file_path} for writing.")

        for class_label, speakers in balanced_speakers.items():
            print(f"Processing class label {class_label} with {len(speakers)} speakers.")

            for speaker_id in speakers:
                audio_files = speaker_data[speaker_id]['files']
                for file_path in audio_files:

                    print(f"Loading audio file {file_path}...")
                    audio, sr = librosa.load(file_path, sr=None)

                    print(f"Extracting segments from {file_path}...")
                    segments = extraction_func(audio, sr, mean=mean, std=std)[:optimum_segments]
                    for i, segment in enumerate(segments):
                        grp_name = f"{speaker_id}_{class_label}_{i}"
                        grp = h5f.create_group(grp_name)
                        grp.create_dataset('audio', data=segment, compression='gzip')
                        grp.attrs['label'] = class_label
                        print(
                            f"Saved segment {i + 1}/{len(segments)} for speaker ID {speaker_id}, class {class_label}.")

    print(f"Data preprocessed and saved to {output_file_path} with balanced speakers across classes.")


######################################################################################################################


# def preprocess_and_save_features(file_paths, labels, output_file_path, augment=False,
#                                  extraction_func=EXTRACTION_FUNCTION, mean=None, std=None):
#
#     output_dir = os.path.dirname(output_file_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     with h5py.File(output_file_path, 'w') as h5f:
#         for i, (file_path, label) in enumerate(zip(file_paths, labels)):
#             print(f"Processing file: {file_path}")
#             audio, sr = librosa.load(file_path, sr=SAMPLERATE)
#
#             if len(audio) > MAX_LENGTH:
#                 audio = audio[:MAX_LENGTH]
#             elif len(audio) < MAX_LENGTH:
#                 padding = MAX_LENGTH - len(audio)
#                 audio = np.pad(audio, (0, padding), 'constant')
#
#             audio_features_padded = extraction_func(audio, sr, mean=mean, std=std)
#
#             # Map multi-class labels to binary labels here
#             if label in [0, 1]:    # Non-depressed
#                 binary_label = 0
#             elif label in [2, 3]:  # Depressed
#                 binary_label = 1
#
#             grp = h5f.create_group(str(i))
#             grp.create_dataset('audio', data=audio_features_padded, compression='gzip')
#             grp.attrs['label'] = binary_label
#
#             if augment:
#                 # Determine the number of augmentations based on the label
#                 if binary_label == 0:
#                     num_augmentations = male_non_depressed_augmentations + female_non_depressed_augmentations
#                 elif binary_label == 1:
#                     num_augmentations = male_depressed_augmentations + female_depressed_augmentations
#
#                 for aug_index in range(num_augmentations):
#                     augmentation_type = random.choice(['noise'])
#                     print(f"Augmentation type: {augmentation_type}")
#
#                     if augmentation_type == 'noise':
#                         augmented_audio = audio_utils.add_noise(audio)
#                     # elif augmentation_type == 'stretch':
#                     #     stretch_rate = random.uniform(0.9, 1.1)
#                     #     augmented_audio = time_stretch(audio, rate=stretch_rate)
#                     # elif augmentation_type == 'shift':
#                     #     n_steps = random.randint(-1, 1)
#                     #     augmented_audio = pitch_shift(audio, sr=sr, n_steps=n_steps)
#
#                     augmented_features = EXTRACTION_FUNCTION(augmented_audio, sr, mean=mean, std=std)
#                     aug_grp = h5f.create_group(f"{i}_aug_{aug_index}")
#                     aug_grp.create_dataset('audio', data=augmented_features, compression='gzip')
#                     aug_grp.attrs['label'] = binary_label
#
#                     print(f"Augmented audio label: {binary_label}")


######################################################################################################################


preprocess_and_save_features(
    train_files,
    train_labels,
    './processed_audio_features/train_features.h5',
    # augment=False,
    extraction_func=EXTRACTION_FUNCTION,
    mean=global_mel_mean,
    std=global_mel_std
)

preprocess_and_save_features(
    dev_files,
    dev_labels,
    './processed_audio_features/dev_features.h5',
    # augment=False,
    extraction_func=EXTRACTION_FUNCTION,
    mean=global_mel_mean,
    std=global_mel_std
)

preprocess_and_save_features(
    test_files,
    test_labels,
    './processed_audio_features/test_features.h5',
    # augment=False,
    extraction_func=EXTRACTION_FUNCTION,
    mean=global_mel_mean,
    std=global_mel_std
)
