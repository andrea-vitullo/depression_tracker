import numpy as np
import librosa
import warnings

import my_config
from utils import audio_utils


def compute_global_stats(files, extraction_func):
    """
    Computes and returns the global mean and standard deviation across all values in a specified extraction
    function, obtained from all provided files.

    Args:
        files (list): A list of paths to audio files.
        extraction_func (function): A function which performs feature extraction on audio data (e.g., computes MFCCs).

    Returns:
        mean (float): The global mean computed across all values in every segment from all files.
        std_dev (float): The global standard deviation computed across all values in every segment from all files.

    Notes:
        This function uses the Welford's online algorithm to compute the mean and variance.
        Extraction function is expected to return a list of 2D arrays
        for a single audio file.
        Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products".
        Technometrics. 4 (3): 419–420. doi:10.2307/1266577. JSTOR 1266577.
    """

    count = 0
    mean = 0.0
    m2 = 0.0

    total_files = len(files)
    print(f"Starting computation of global stats for {total_files} files.")

    for i, file_path in enumerate(files):
        print(f"Processing file {i + 1}/{total_files}: {file_path}")
        try:
            audio, sr = librosa.load(file_path, sr=None)
            segments = extraction_func(audio, sr)

            # Flatten segments and iterate over each value
            for value in np.concatenate(segments).ravel():
                count += 1

                # Update mean and m2
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                m2 += delta * delta2

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Compute standard deviation
    variance_n = m2 / (count if count > 0 else 1)
    std_dev = np.sqrt(variance_n)

    print(f"Global mean: {mean}, Global std: {std_dev}")

    return mean, std_dev


def extract_raw_audio_segments(audio, target_length=61440, mean=0, std=1):
    """
    Extracts segments from the given raw audio data and standardizes each segment.

    This function divides a given audio signal into multiple segments of equal length. The target segment length
    is defined by the target_length parameter. Each segment is then standardized using the provided mean and standard
    deviation, and reshaped into (segment_length, 1) before being stored in the list of segments.

    Args:
        audio (numpy.ndarray): The raw audio data to segment.
        target_length (int, optional): The desired length of each segment. Default is 61440.
        mean (float, optional): The mean value used for standardizing the audio data. Default is 0.
        std (float, optional): The standard deviation used for standardizing the audio data. Default is 1.

    Returns:
        raw_audio_segments (list): List of the extracted raw audio segments.
    """

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // target_length

    raw_audio_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * target_length
        end_sample = start_sample + target_length

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Standardize audio data
        segment = audio_utils.standardization(segment, mean, std)

        raw_audio_segments.append(segment[:, np.newaxis])

    if raw_audio_segments:
        print(f"One raw audio segment shape: {raw_audio_segments[0].shape} (Expected: (61440, 1))")

    return raw_audio_segments


def extract_mfcc_segments(audio, sr,
                          n_mels=my_config.MFCC_MELS,
                          n_fft=my_config.N_FTT,
                          hop_length=my_config.MFCC_HOP_LENGTH,
                          n_mfcc=my_config.N_MFCC, count_segments=False):
    """
    This function splits the audio into segments and computes the Mel Frequency Cepstral Coefficients (MFCCs)
    for each segment.

    Args:
        audio (numpy.ndarray): The raw waveform of the audio signal.
        sr (int): The sampling rate of the audio.
        n_mels (int, optional): Number of Mel filters to use. Default is 256.
        n_fft (int, optional): Length of the FFT window. Default is 1024.
        hop_length (int, optional): Amount of steps to move the analysis window for each Fourier Transform.
                                    Default is 128.
        n_mfcc (int, optional): Number of MFCCs to return. Default is 13.
        count_segments (bool, optional): If true, it counts the number of segments that can be extracted

    Returns:
        num_segments (int): If count_segments is True, it returns the number of segments that can be extracted from
                            the audio.
        mfcc_segments (list): If count_segments is False, it returns a list of the MFCC matrices for each segment
                              extracted from the audio.

    Notes:
        - The length of each segment (245,760 samples) is calculated to ensure that the resulting MFCC for each
          segment forms a uniform matrix of size 480 x n_mfcc.
        - Each segment is extracted sequentially from the input audio, and for each segment, an MFCC is computed
          and stored.
        - If the resulting MFCC for a segment is shorter than 480 frames, zero-padding is applied; if it's longer,
          the excess frames are discarded.
    """

    # 245760 is the length of each segment in samples to fit exactly 480 frames
    # 61,440 samples to fit 120 frames, 122,880 to fit 240 and so on...
    segment_length_samples = 245760

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    if count_segments:
        return num_segments

    mfcc_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Compute the MFCCs for the current segment
        mfccs = librosa.feature.mfcc(segment, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        mfccs_transposed = mfccs.T

        # Trim the last time frame if mfcc shape exceeds the expected frame count
        if mfccs_transposed.shape[0] > 480:
            mfccs_transposed = mfccs_transposed[:480, :]
        elif mfccs_transposed.shape[0] < 480:
            pad_width = ((0, 480 - mfccs_transposed.shape[0]), (0, 0))  # Padding applies only to time axis
            mfccs_transposed = np.pad(mfccs_transposed, pad_width, mode='constant')

        # Append the processed MFCC segment to the list
        mfcc_segments.append(mfccs_transposed)

    # Check the shape of one segment to verify the dimensions
    if mfcc_segments:
        print(f"One MFCC segment shape: {mfcc_segments[0].shape} (Expected: (120, {n_mfcc}))")

    return mfcc_segments


def extract_mel_segments(audio, sr,
                         n_mels=my_config.MEL_SPEC_MELS,
                         n_fft=my_config.N_FTT,
                         hop_length=my_config.MEL_HOP_LENGTH):
    """
    This function segments the audio and extracts the logarithmic Mel-spectrograms for each segment.

    Args:
        audio (numpy.ndarray): The array containing audio samples.
        sr (int): The sampling rate of the audio.
        n_mels (int, optional): Number of Mel bands to generate. Default is 256.
        n_fft (int, optional): The FFT window size. Default is 1024.
        hop_length (int, optional): The number of samples between successive frames. Default is 512.

    Returns:
        logmel_segments (List[numpy.ndarray]): A list of logarithmic Mel-spectrogram matrices for each extracted
        audio segment.

    Notes:
        - Each audio frame is extracted as a segment of fixed size (245760 samples) to match the Mel-spectrogram shape
          of (480, n_mels).
        - This size is calculated as it fits exactly 480 frames to ensure a consistent input size.
          The count of 480 is arbitrary and could be changed depending on your use case.
        - After clearly segmenting the audio, each segment is processed independently, converting the segment to a
          Mel-spectrogram and then converting the power spectrogram to dB units.
        - If the Mel-spectrogram exceeds the expected frame count of 480, it is trimmed to fit.
    """

    # 245760 is the length of each segment in samples to fit exactly 480 frames
    # 61,440 samples to fit 120 frames, 122,880 to fit 240 and so on...
    segment_length_samples = 245760

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    logmel_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Compute the mel spectrogram for the current segment
        melspectrogram = librosa.feature.melspectrogram(segment, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                        hop_length=hop_length)

        # Convert the power spectrogram to decibel units
        logmelspec = librosa.power_to_db(melspectrogram)

        logmelspec_transposed = logmelspec.T

        # Trim the last time frame if the spectrogram shape exceeds the expected frame count
        if logmelspec_transposed.shape[0] > 480:
            logmelspec_transposed = logmelspec_transposed[:480, :]

        # Append the processed log-mel spectrogram segment to the list
        logmel_segments.append(logmelspec_transposed)

    # Check the shape of one segment to verify the dimensions
    if logmel_segments:
        print(f"One log-mel segment shape: {logmel_segments[0].shape} (Expected: (120, 256))")

    return logmel_segments


def extract_spectrogram_segments(audio,
                                 n_fft=my_config.N_FTT,
                                 hop_length=my_config.SPECTROGRAM_HOP_LENGTH):
    """
    This function segments the audio and computes logarithmic spectrograms for each segment.

    Args:
        audio (numpy.ndarray): Input audio signal as a numpy array.
        n_fft (int, optional): Length of the FFT window. Default is 1024.
        hop_length (int, optional): Number of audio samples between successive frames. Default is 512.

    Returns:
        spectrogram_segments (List[numpy.ndarray]): A list containing the logarithmic spectrogram of each segmented
        frame of the audio.

    Notes:
        - The input audio is divided into segments of fixed length (here, 245760 samples). This size is set to fit
          exactly 480 spectrogram frames for each audio segment.
        - For each of these segments, the Short-Time Fourier Transform (STFT) is computed, and then converted to a
          spectrogram by squaring the magnitude of the complex STFT.
        - The power spectrogram is then converted into decibel units to form a logarithmic spectrogram.
        - If the computed logarithmic spectrogram exceeds the desired 480-frame size, it's trimmed accordingly.
        - Finally, all spectrogram segments are returned as a list.
    """

    # 245760 is the length of each segment in samples to fit exactly 480 frames
    # 61,440 samples to fit 120 frames, 122,880 to fit 240 and so on...
    segment_length_samples = 245760

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    spectrogram_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Compute the STFT for the current segment
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)

        # Compute the spectrogram, which is the squared magnitude of each component in STFT matrix
        spectrogram = np.abs(stft)**2

        # Convert the power spectrogram to decibel units
        logspectrogram = librosa.power_to_db(spectrogram)

        logspectrogram_transposed = logspectrogram.T

        if logspectrogram_transposed.shape[0] > 480:
            logspectrogram_transposed = logspectrogram_transposed[:480, :]

        # Append the processed log-spectrogram segment to the list
        spectrogram_segments.append(logspectrogram_transposed)

    # Check the shape of one segment to verify the dimensions
    if spectrogram_segments:
        # n_fft//2 + 1 is the number of unique STFT bins
        print(f"One spectrogram segment shape: {spectrogram_segments[0].shape} (Expected: (120, {n_fft // 2 + 1}))")

    return spectrogram_segments


def extract_chroma_segments(audio, sr,
                            n_fft=my_config.N_FTT,
                            hop_length=my_config.CHROMA_HOP_LENGTH):
    """
    This function segments the audio and computes Chroma features for each segment.

    Args:
        audio (numpy.ndarray): Input audio signal.
        sr (int): The sampling rate of the audio.
        n_fft (int, optional): FFT window size. Default is 1024.
        hop_length (int, optional): The number of samples between successive frames. Default is 128.

    Returns:
        chroma_segments (List[numpy.ndarray]): A list containing the Chroma feature matrix for each segment of
        the audio.

    Notes:
        - The audio is divided into segments of fixed length (245760 samples). This size is chosen to match exactly
          480 chroma frames per segment.
        - The chroma features, often used in music information retrieval, represent the energy distribution across
          different pitch classes.
        - For each segment, the Chroma features are computed and then transposed to match the expected shape.
        - If the resulting chroma frames exceed or are less than the expected 480 frames, adjustments are made by
          either trimming or padding the frames with zeros.
    """

    # 245760 is the length of each segment in samples to fit exactly 480 frames
    # 61,440 samples to fit 120 frames, 122,880 to fit 240 and so on...
    segment_length_samples = 245760

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    chroma_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # A chromagram is a representation of the short-term energy within different chroma bands (essentially pitches),
        # which is often used in musical audio analysis. However, the computation of a chromagram can sometimes result
        # in complex numbers, which results in a np.ComplexWarning when further computations are performed on the
        # chromagram. The use of the warnings.catch_warnings() context manager is to prevent this np.ComplexWarning
        # from being shown.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=np.ComplexWarning)

            chroma = librosa.feature.chroma_stft(segment, sr=sr, n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_chroma=my_config.N_CHROMA)

            chroma_transposed = chroma.T

            # Trim the last time frame if chroma shape exceeds the expected frame count
            if chroma_transposed.shape[0] > 480:
                chroma_transposed = chroma_transposed[:480, :]
            elif chroma_transposed.shape[0] < 480:
                pad_width = ((0, 480 - chroma_transposed.shape[0]), (0, 0))  # Padding applies only to time axis
                chroma_transposed = np.pad(chroma_transposed, pad_width, mode='constant')

            # Append the processed chroma segment to the list
            chroma_segments.append(chroma_transposed)

    # Check the shape of one segment to verify the dimensions
    if chroma_segments:
        print(f"One chroma segment shape: {chroma_segments[0].shape} (Expected: (120, 12))")

    return chroma_segments
