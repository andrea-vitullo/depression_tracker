import unittest
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import soundfile as sf
import random

from my_config import *
from features_extractors import extract_raw_audio, extract_mfcc_segments, extract_logmel, extract_logmel_segments, extract_chroma_segments
from utils.utils import compute_global_stats_from_test_data


class FeatureExtractionTest(unittest.TestCase):
    def test_raw_audio_extraction(self):
        # Get a list of all .wav files in the audio test directory
        audio_files = [f for f in os.listdir(AUDIO_TEST_DIRECTORY) if f.endswith('.wav')]

        # Loop through each audio file in the list
        for audio_file in audio_files:
            # Create the full file path by joining the directory path and audio file name
            audio_file_path = os.path.join(AUDIO_TEST_DIRECTORY, audio_file)

            # Load each audio file individually
            audio, sr = sf.read(audio_file_path)

            # Define target_length based on your `NSEG` and `H` values
            target_length = NSEG * H

            # Extract and standardize audio
            audio = extract_raw_audio(audio, sr, target_length)

            # Compute time array
            time = np.arange(0, len(audio)) / sr

            # Create output directory if not exists
            output_directory = os.path.join(AUDIO_TEST_DIRECTORY, 'raw_audio_extraction')
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Plot amplitudes over time
            plt.figure(figsize=(12, 4))
            plt.plot(time, audio)
            plt.title('Audio signal over time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')

            # Save the plot to a file in the output directory with the same basename as the current audio file
            plt.savefig(os.path.join(output_directory, f'{os.path.splitext(audio_file)[0]}_audio_plot.png'))
            plt.close()

    def test_mfcc_extraction(self):
        # Get a list of all .wav files in the audio test directory
        audio_files = [f for f in os.listdir(AUDIO_TEST_DIRECTORY) if f.endswith('.wav')]

        # Loop through each audio file in the list
        for audio_file in audio_files:
            # Create the full file path by joining the directory path and audio file name
            audio_file_path = os.path.join(AUDIO_TEST_DIRECTORY, audio_file)

            # Load each audio file individually
            audio, sr = sf.read(audio_file_path)

            # Extract MFCC segments and transpose
            mfcc_segments = extract_mfcc_segments(audio, sr)

            # For simplicity, let's only plot the first segment
            mfcc_segment = mfcc_segments[0]

            # Create MFCC output directory if not exists
            mfcc_output_directory = os.path.join(AUDIO_TEST_DIRECTORY, 'mfcc_extraction')
            if not os.path.exists(mfcc_output_directory):
                os.makedirs(mfcc_output_directory)

            norm = Normalize(vmin=np.min(mfcc_segment), vmax=np.max(mfcc_segment))

            # Plot the MFCC features as Heatmap and save the plot
            plt.figure(figsize=(10, 4))
            plt.imshow(mfcc_segment.T, origin='lower', aspect='auto', cmap='inferno', norm=norm)
            plt.title('MFCCs')
            plt.ylabel('MFCC coefficients')
            plt.xlabel('Time (frames)')
            plt.colorbar()

            plt.savefig(os.path.join(mfcc_output_directory, f'{os.path.splitext(audio_file)[0]}_mfcc_plot.png'))
            plt.close()

    def test_logmel_extraction(self):

        global_mean, global_std = compute_global_stats_from_test_data(AUDIO_TEST_DIRECTORY)

        # Get a list of all .wav files in the audio test directory
        audio_files = [f for f in os.listdir(AUDIO_TEST_DIRECTORY) if f.endswith('.wav')]

        # Loop through each audio file in the list
        for audio_file in audio_files:
            # Create the full file path by joining the directory path and audio file name
            audio_file_path = os.path.join(AUDIO_TEST_DIRECTORY, audio_file)

            # Load each audio file individually
            audio, sr = sf.read(audio_file_path)

            # Extract LogMel features
            logmel_features = extract_logmel(audio, sr, mean=global_mean, std=global_std)

            # Remove the last dimension for plotting
            logmel_features_2d = logmel_features.squeeze(-1)

            # Create LogMel output directory if not exists
            logmel_output_directory = os.path.join(AUDIO_TEST_DIRECTORY, 'logmel_extraction')
            if not os.path.exists(logmel_output_directory):
                os.makedirs(logmel_output_directory)

            # Plot the LogMel features as Heatmap and save the plot
            plt.figure(figsize=(10, 4))
            plt.imshow(logmel_features_2d.T, origin='lower', aspect='auto', cmap='viridis')
            plt.title('LogMel Spectrogram')
            plt.ylabel('Mel Filterbanks')
            plt.xlabel('Time (frames)')
            plt.colorbar()

            plt.savefig(os.path.join(logmel_output_directory, f'{os.path.splitext(audio_file)[0]}_logmel_plot.png'))
            plt.close()

    def test_logmel_segments_extraction(self):

        global_mean, global_std = compute_global_stats_from_test_data(AUDIO_TEST_DIRECTORY)

        # Get a list of all .wav files in the audio test directory
        audio_files = [f for f in os.listdir(AUDIO_TEST_DIRECTORY) if f.endswith('.wav')]

        # Randomly select one audio file from the list
        random_audio_file = random.choice(audio_files)
        audio_file_path = os.path.join(AUDIO_TEST_DIRECTORY, random_audio_file)

        # Load the selected audio file
        audio, sr = sf.read(audio_file_path)

        # Extract LogMel segments
        logmel_segments = extract_logmel_segments(audio, sr, mean=global_mean, std=global_std)

        # Select 5 continuous segments for visualization
        selected_segments = logmel_segments[:5]

        # Create LogMel segments output directory if not exists
        logmel_segments_output_directory = os.path.join(AUDIO_TEST_DIRECTORY, 'logmel_segments_extraction')
        if not os.path.exists(logmel_segments_output_directory):
            os.makedirs(logmel_segments_output_directory)

        # Plot each of the selected LogMel segments
        for i, segment in enumerate(selected_segments):
            plt.figure(figsize=(10, 4))
            plt.imshow(segment.T, origin='lower', aspect='auto', cmap='viridis')
            plt.title(f'LogMel Spectrogram Segment {i + 1}')
            plt.ylabel('Mel Filterbanks')
            plt.xlabel('Time (frames)')
            plt.colorbar()

            plt.savefig(os.path.join(logmel_segments_output_directory,
                                     f'{os.path.splitext(random_audio_file)[0]}_logmel_segment_{i + 1}.png'))
            plt.close()

    def test_chroma_extraction(self):
        # Get a list of all .wav files in the audio test directory
        audio_files = [f for f in os.listdir(AUDIO_TEST_DIRECTORY) if f.endswith('.wav')]

        # Loop through each audio file in the list
        for audio_file in audio_files:
            # Create the full file path by joining the directory path and audio file name
            audio_file_path = os.path.join(AUDIO_TEST_DIRECTORY, audio_file)

            # Load each audio file individually
            audio, sr = sf.read(audio_file_path)

            # Extract chroma segments and transpose
            chroma_segments = extract_chroma_segments(audio, sr)

            # For simplicity, let's only plot the first segment
            chroma_segment = chroma_segments[0]

            # Create chroma output directory if not exists
            chroma_output_directory = os.path.join(AUDIO_TEST_DIRECTORY, 'chroma_extraction')
            if not os.path.exists(chroma_output_directory):
                os.makedirs(chroma_output_directory)

            norm = Normalize(vmin=np.min(chroma_segment), vmax=np.max(chroma_segment))

            # Plot the chroma features as Heatmap and save the plot
            plt.figure(figsize=(10, 4))
            plt.imshow(chroma_segment.T, origin='lower', aspect='auto', cmap='coolwarm', norm=norm)
            plt.title('Chroma Features')
            plt.ylabel('Pitch class')
            plt.xlabel('Time (frames)')
            plt.colorbar()

            plt.savefig(os.path.join(chroma_output_directory, f'{os.path.splitext(audio_file)[0]}_chroma_plot.png'))
            plt.close()


if __name__ == '__main__':
    unittest.main()
