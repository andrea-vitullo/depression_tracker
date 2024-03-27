import unittest
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import soundfile as sf

from my_config import *
from features_extractors import extract_raw_audio, extract_mfcc, extract_logmel


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

    def test_mfcc_extraction(self):
        # Get a list of all .wav files in the audio test directory
        audio_files = [f for f in os.listdir(AUDIO_TEST_DIRECTORY) if f.endswith('.wav')]

        # Loop through each audio file in the list
        for audio_file in audio_files:
            # Create the full file path by joining the directory path and audio file name
            audio_file_path = os.path.join(AUDIO_TEST_DIRECTORY, audio_file)

            # Load each audio file individually
            audio, sr = sf.read(audio_file_path)

            # Extract MFCC features
            mfcc_features = extract_mfcc(audio, sr)

            # Remove the last dimension for plotting
            mfcc_features_2d = mfcc_features.squeeze(-1)

            # Create MFCC output directory if not exists
            mfcc_output_directory = os.path.join(AUDIO_TEST_DIRECTORY, 'mfcc_extraction')
            if not os.path.exists(mfcc_output_directory):
                os.makedirs(mfcc_output_directory)

            norm = Normalize(vmin=np.min(mfcc_features_2d), vmax=np.max(mfcc_features_2d))

            # Plot the MFCC features as Heatmap and save the plot
            plt.figure(figsize=(10, 4))
            plt.imshow(mfcc_features_2d.T, origin='lower', aspect='auto', cmap='inferno', norm=norm)
            plt.title('MFCCs')
            plt.ylabel('MFCC coefficients')
            plt.xlabel('Time (frames)')
            plt.colorbar()
            plt.savefig(os.path.join(mfcc_output_directory, f'{os.path.splitext(audio_file)[0]}_mfcc_plot.png'))

    def test_logmel_extraction(self):
        # Get a list of all .wav files in the audio test directory
        audio_files = [f for f in os.listdir(AUDIO_TEST_DIRECTORY) if f.endswith('.wav')]

        # Loop through each audio file in the list
        for audio_file in audio_files:
            # Create the full file path by joining the directory path and audio file name
            audio_file_path = os.path.join(AUDIO_TEST_DIRECTORY, audio_file)

            # Load each audio file individually
            audio, sr = sf.read(audio_file_path)

            # Extract LogMel features
            logmel_features = extract_logmel(audio, sr)

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

if __name__ == '__main__':
    unittest.main()
