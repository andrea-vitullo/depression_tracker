import os
import subprocess
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import numpy as np

import my_config


def audio_converter(audio):
    """
    Converts audio file into mono-16kHz, and overwrites it

    Args:
        audio: Audio file to be converted
    Return:
         None
    """
    waveform, sample_rate = sf.read(audio)

    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())
        print(f"Found stereo waveform: {waveform}\nConverted to mono.")

    if sample_rate != my_config.SAMPLERATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=my_config.SAMPLERATE)
        print(f"Wrong sample rate for file: {waveform}\nConverted to 16kHz.\n")

    sf.write(audio, waveform, my_config.SAMPLERATE)


def audio_formatter():
    """
    Check for file format of audio files and convert them to mono-16kHz

    Args:
        -
    Returns:
        None
    """
    for path, directories, files in os.walk(my_config.DIRECTORY):
        for audio in files:
            if audio.endswith(my_config.FILE_FORMAT):
                full_audio_path = os.path.join(path, audio)
                print(f"Checking {full_audio_path}...")
                audio_converter(full_audio_path)


def diarisation(t_audio, d_frame, d_path):
    """
    Perform diarisation on participant audio, and then saves the sliced samples

    Args:
        t_audio: Temporary audio audiosegment object
        d_frame: Participant part of the transcript in a 2D dataframe
        d_path: Path to the temporary audio file
    Return:
         None
    """
    if d_frame is not None and t_audio is not None:
        for _, row in d_frame.iterrows():
            t1 = row.get('start_time', None)
            t2 = row.get('stop_time', None)
            if t1 is not None and t2 is not None:
                t1 *= 1000
                t2 *= 1000
                newaudio = t_audio[t1:t2]
                newaudio.export(os.path.join(d_path, f'{t1}_{t2}_SPLIT.wav'), format=my_config.FILE_FORMAT)
            else:
                print("Data row does not contain valid 'start_time' / 'stop_time'!")


def get_audio_lengths():
    """
    Get audio lengths in seconds and plot them in a figure

    Args:
        -
    Returns:
        None
    """

    audio_lengths = {}

    for path, directories, files in os.walk(my_config.DIRECTORY):
        for audio in files:
            if audio.endswith(my_config.FINAL_FORMAT):
                full_audio_path = os.path.join(path, audio)
                waveform, sample_rate = librosa.load(full_audio_path)
                audio_length_sec = len(waveform) / sample_rate
                audio_lengths[audio] = audio_length_sec

    names = list(audio_lengths.keys())
    values = list(audio_lengths.values())

    plt.subplots(figsize=(80, 20))
    plt.bar(names, values)
    plt.title("Lengths of audio files")
    plt.xlabel("File names")
    plt.ylabel("Length (seconds)")
    plt.xticks(names, rotation='vertical')
    plt.show()


def get_raw_audio(path, audio):
    """
    Generate the input and output filename for raw audio processing.

    Args:
        path (str): The directory path where the audio file is located.
        audio (str): The filename of the audio file to be processed.

    Returns:
        tuple: A tuple containing the full paths of the input and output audio files.
    """

    cleaned = audio.replace(my_config.FILE_FORMAT, my_config.CLEANED_FORMAT)
    ip = os.path.join(path, audio)
    op = os.path.join(path, cleaned)
    print(f'Found audio file: {ip}')

    return ip, op


def remove_noise(input_audio, output_audio):
    """
    Runs a command that uses FFmpeg to apply several audio filters
    to the input audio file (-i <input_audio>) and writes the result to the
    output file.

    The audio filters applied are:
    - anlmdn: Adaptive non-local means denoising. s, p, r, and m are parameters
              to control the intensity of noise reduction.
    - highpass: Filters out audio frequencies below a particular threshold (150Hz).
    - lowpass: Filters out audio frequencies above a particular threshold (4000Hz).

    Args:
        input_audio (str): The full path of the audio file to be denoised.
        output_audio (str): The full path where the denoised audio file will be saved.

    Returns:
        None.

    Raises:
        Exception: An exception is raised and the error message printed
                   if the subprocess call to FFmpeg fails.
    """

    command = ['ffmpeg', '-i', input_audio, '-af', 'anlmdn=s=0.001:p=0.002:r=0.004:m=12, '
                                                   'highpass=f=150, '
                                                   'lowpass=f=4000',
               output_audio]

    print(f"Processing: {input_audio}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return

    print(f'File processed and saved as: {output_audio}\n')


def normalise():
    """
    Normalize audio files in specified directory

    Args:
        -
    Returns:
        None
    """

    target_sample_rate = my_config.SAMPLERATE

    for path, directories, files in os.walk(my_config.DIRECTORY):
        for audio in files:
            if audio.endswith(my_config.FINAL_FORMAT):
                full_audio_path = os.path.join(path, audio)

                waveform, sample_rate = librosa.load(full_audio_path, sr=None)
                waveform = waveform / np.max(np.abs(waveform))
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
                sf.write(full_audio_path, waveform, target_sample_rate)


def min_max_normalization(data):
    """
    From a set of data, normalise the data to be between 0 and 1 (inclusive)

    Args:
        data: The data to be processed

    Returns:
        normalised_data: Output normalised data with min 0 and max 1
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalised_data = (data - min_val) / (max_val - min_val)

    return normalised_data


def standardization(data, mean, std):
    """
    From a set of data, normalise the data using the mean and the standard
    deviation to obtain 0 mean and standard deviation of 1

    Args:
        data: The data to be processed
        mean: The mean of the data
        std: The standard deviation of the data

    Returns:
        normalised_data: Output normalised data with mean 0 and standard
                         deviation of 1
    """
    normalised_data = (data-mean) / std

    return normalised_data


def add_noise(data, noise_factor=0.005):
    """
    Adds Gaussian noise to the input data.

    Args:
    data (numpy.ndarray): The input data array to which noise is to be added.
    noise_factor (float, optional): The factor to multiply the randomly generated noise by. Defaults to 0.005.

    Returns:
    numpy.ndarray: The "noisy" data obtained by adding scaled Gaussian noise to the original data.
    """

    noise = np.random.randn(len(data))
    return data + noise_factor * noise
