import os
import glob
import librosa
import numpy as np
import soundfile as sf

import my_config


def extract_non_silent(audio, threshold=1e-10):
    """
    Extracts non-silent segments from an audio signal.

    Args:
        audio: The audio signal as a numpy array.
        threshold: Amplitude below which a sample is considered silent.

    Returns:
        non_silent_segments: A list of non-silent segments. Each segment is a list of audio samples.
    """

    non_silent_segments = []
    current_segment = []
    is_silent = lambda smpl: abs(smpl) < threshold

    for sample in audio:
        if is_silent(sample):
            if current_segment:
                non_silent_segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(sample)

    # Add the last segment if it's non-silent
    if current_segment:
        non_silent_segments.append(current_segment)

    return non_silent_segments


def process_audio_file(file_path):
    """
    Process a single audio file by replacing silent segments with non-silent ones.
    Creates a new audio file with silence removed.

    Args:
        file_path: The path to the audio file to process.

    Returns:
        None
    """

    audio, sr = librosa.load(file_path, sr=None)
    non_silent_audio_segments = extract_non_silent(audio, sr)

    # Combine non-silent segments into a single array
    combined_segments = np.concatenate([np.array(segment) for segment in non_silent_audio_segments])

    output_file_path = file_path.replace('_Final.wav', '_NoSilence.wav')
    sf.write(output_file_path, combined_segments, sr)

    print(f"Processed and saved: {output_file_path}")


def process_files_in_folder(folder_path):
    """
    Process all audio files in a folder and its sub-folders. Audio files are identified by the '_Final.wav' suffix.
    For each audio file, a new file is created with silent segments removed.

    Args:
        folder_path: Path to the folder containing audio files.

    Returns:
        None
    """

    search_pattern = os.path.join(folder_path, '**', '*_Final.wav')

    for wav_file in glob.glob(search_pattern, recursive=True):
        print(f"Processing {wav_file}...")
        process_audio_file(wav_file)


main_folder_path = my_config.DIRECTORY
process_files_in_folder(main_folder_path)
