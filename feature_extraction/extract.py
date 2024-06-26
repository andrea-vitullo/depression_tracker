import os
import librosa
import h5py
import re
from collections import defaultdict

import my_config
from utils import utils
import features_extractors


# EXTRACTION FUNCTIONS
# based on extraction type to perform import from features_extractors
EXTRACTION_FUNCTIONS = {
    'mfcc': features_extractors.extract_mfcc_segments,
    'chroma': features_extractors.extract_chroma_segments,
    'logmel': features_extractors.extract_mel_segments,
    'spectrogram': features_extractors.extract_spectrogram_segments
}


# Load the data
train_files, train_labels = utils.load_files_labels(my_config.AUDIO_TRAIN_DIR)
dev_files, dev_labels = utils.load_files_labels(my_config.AUDIO_DEV_DIR)
test_files, test_labels = utils.load_files_labels(my_config.AUDIO_TEST_DIR)


# Define datasets
datasets = {
    'train': (train_files, train_labels),
    'dev': (dev_files, dev_labels),
    'test': (test_files, test_labels),
}


# Compute global stats for each type of feature across training set
global_stats = {}
for feature_name, extraction_func in EXTRACTION_FUNCTIONS.items():
    print(f"Computing global stats for {feature_name} features.")
    mean, std = features_extractors.compute_global_stats(train_files, extraction_func)
    global_stats[feature_name] = {'mean': mean, 'std': std}


def preprocess_and_save_features(file_paths, dataset_labels, out_file_path, extraction_function,
                                 speakers_per_class=None, max_segments=10, group_mean=0, group_std=1):
    """
    Preprocess given audio files and extract the features of each segment. Then, it saves those features to HDF5 file.

    This function scans the given file paths, extracts the features from each audio segment using the provided
    extraction function, and then saves those features into a HDF5 file at the given output path.
    The function also maintains a balance of speakers per class, thus ensuring that the data does not become skewed
    towards any class.

    Args:
        file_paths (list): List of file paths of the audio files.
        dataset_labels (list): Corresponding labels for each of the audio files.
        out_file_path (str): The output path for saving the preprocessed data as a HDF5 file.
        extraction_function (callable): Function used to extract features from the audio data.
        speakers_per_class (int, optional): The desired number of speakers per class. If not provided, the minimum
                                            number of speakers across all classes will be used.
        max_segments (int, optional): Maximum number of segments to process per audio file.
        group_mean (float, optional): Normalization factor (mean value) used during feature extraction. Default is 0.
        group_std (float, optional): Normalization factor (standard deviation value) used during feature extraction.
                                     Default is 1.

    Returns:
        None. The extracted features are written to a HDF5 file.
    """

    # Initialize counters for depressed/non-depressed segments
    label_counters = defaultdict(int)

    print(f"Creating directory {os.path.dirname(out_file_path)} if it doesn't exist...")
    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

    # Map speaker IDs to their audio files and labels
    speaker_data = {}

    print("Mapping speaker IDs to their files and labels...")
    for path, label in zip(file_paths, dataset_labels):
        speaker_id = re.findall(r"(\d+)_NoSilence.wav", path)[0]
        if speaker_id not in speaker_data:
            speaker_data[speaker_id] = {'files': [], 'label': label}
            print(f"Speaker ID {speaker_id} added to speaker_data with label {label}.")

        speaker_data[speaker_id]['files'].append(path)

    # Calculate the number of speakers per class only if it's not provided
    if speakers_per_class is None:
        class_counts = {label: 0 for label in set(dataset_labels)}
        for speaker in speaker_data.values():
            class_counts[speaker['label']] += 1
        speakers_per_class = min(class_counts.values())
        print(f"Speakers per class determined: {speakers_per_class}")
    else:
        print(f"Speakers per class specified: {speakers_per_class}")

    # Select a balanced set of speakers for each class
    print("Selecting a balanced set of speakers for each class...")
    balanced_speakers = {label: [] for label in set(dataset_labels)}
    for speaker_id, data in speaker_data.items():
        if len(balanced_speakers[data['label']]) < speakers_per_class:
            balanced_speakers[data['label']].append(speaker_id)
            print(f"Speaker ID {speaker_id} with label {data['label']} selected.")

    with h5py.File(out_file_path, 'w') as h5f:
        print(f"Opened {out_file_path} for writing.")

        for class_label, speakers in balanced_speakers.items():
            print(f"Processing class label {class_label} with {len(speakers)} speakers.")

            for speaker_id in speakers:
                audio_files = speaker_data[speaker_id]['files']
                for file_path in audio_files:

                    print(f"Loading audio file {file_path}...")
                    audio, sr = librosa.load(file_path, sr=None)

                    print(f"Extracting segments from {file_path}...")
                    all_segments = extraction_function(audio, sr, mean=group_mean, std=group_std)

                    # Skip the first segment and limit the number of segments
                    segments = all_segments[1:max_segments+1]

                    for i, segment in enumerate(segments):
                        grp_name = f"{speaker_id}_{class_label}_{i}"
                        grp = h5f.create_group(grp_name)
                        grp.create_dataset('audio', data=segment, compression='gzip')
                        grp.attrs['label'] = class_label

                        label_counters[class_label] += 1

                        print(f"Saved segment {i + 1}/{len(segments)} "
                              f"for file {os.path.basename(file_path)}, label {label}.")

    print(f"Data preprocessed and saved to {out_file_path}.\n")


# Process and save features for each dataset and extraction function
for dataset_name, (files, labels) in datasets.items():
    for feature_name, extraction_func in EXTRACTION_FUNCTIONS.items():

        output_file_path = (f'/Users/andreavitullo/Desktop/Python/final_project/'
                            f'processed_audio_features/{dataset_name}_{feature_name}.h5')
        mean, std = global_stats[feature_name]['mean'], global_stats[feature_name]['std']
        preprocess_and_save_features(
            files,
            labels,
            output_file_path,
            max_segments=10,
            extraction_function=extraction_func,
            speakers_per_class=None,
            group_mean=mean,
            group_std=std
        )
