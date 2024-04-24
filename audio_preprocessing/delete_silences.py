import os
import glob
import librosa
import numpy as np
import soundfile as sf  # Import the soundfile library


def extract_non_silent(audio, sr, threshold=1e-10):
    """Extracts non-silent segments from an audio signal."""
    non_silent_segments = []
    current_segment = []
    is_silent = lambda sample: abs(sample) < threshold

    for sample in audio:
        if is_silent(sample):
            if current_segment:  # End of a non-silent segment
                non_silent_segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(sample)

    # Add the last segment if it's non-silent
    if current_segment:
        non_silent_segments.append(current_segment)

    return non_silent_segments


def process_audio_file(file_path):
    # Load your audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Extract non-silent segments
    non_silent_audio_segments = extract_non_silent(audio, sr)

    # Combine non-silent segments into a single array
    combined_segments = np.concatenate([np.array(segment) for segment in non_silent_audio_segments])

    # Saving the processed audio back to a file using soundfile library
    output_file_path = file_path.replace('_Final.wav', '_NoSilence.wav')
    sf.write(output_file_path, combined_segments, sr)

    print(f"Processed and saved: {output_file_path}")


def process_files_in_folder(main_folder_path):
    # Refine the search pattern to match your naming convention
    search_pattern = os.path.join(main_folder_path, '**', '*_Final.wav')

    for wav_file in glob.glob(search_pattern, recursive=True):
        print(f"Processing {wav_file}...")
        process_audio_file(wav_file)


# Replace this with the path to your main folder containing subfolders and WAV files
main_folder_path = '/Users/andreavitullo/Desktop/DATABASE_TEST'
process_files_in_folder(main_folder_path)
