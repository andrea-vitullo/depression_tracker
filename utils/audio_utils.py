import os
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

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

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        print(f"Wrong sample rate for file: {waveform}\nConverted to 16kHz.\n")

    sf.write(audio, waveform, 16000)


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

    audio_lengths = []

    for path, directories, files in os.walk(my_config.DIRECTORY):
        for audio in files:
            if audio.endswith(my_config.FINAL_FORMAT):
                full_audio_path = os.path.join(path, audio)
                waveform, sample_rate = librosa.load(full_audio_path)
                audio_length_sec = len(waveform) / sample_rate
                audio_lengths.append(audio_length_sec)

    plt.hist(audio_lengths, bins=50)
    plt.title("Distribution of audio lengths")
    plt.xlabel("Length (seconds)")
    plt.ylabel("Number of audio files")
    plt.show()
