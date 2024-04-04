import numpy as np
import scipy.signal
import librosa


def calculate_formants(y, sr, order):
    """
    Function to calculate formants of a window
    :param y: audio time series
    :param sr: sampling rate
    :param order: the order for linear predictive coding
    :return: array of formants
    """
    # Pre-emphasis
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Compute LPC.
    lpc = librosa.lpc(y, order)
    rts = np.roots(lpc)

    # Get roots.
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies and bandwidth
    frqs = sorted(angz * (sr / (2 * np.pi)))

    return frqs


def calculate_vowel_space(audio_path, window_length, hop_length, formant_order=4, median_filter_order=5):
    y, sr = librosa.load(audio_path)
    num_samples = len(y)
    window_length_samples = int(window_length * sr)
    hop_length_samples = int(hop_length * sr)

    min_f1 = np.inf
    max_f1 = -np.inf
    min_f2 = np.inf
    max_f2 = -np.inf

    for i in range(0, num_samples - window_length_samples + 1, hop_length_samples):
        window = y[i: i + window_length_samples]
        formants = calculate_formants(window, sr, formant_order)
        if len(formants) >= 2:  # Checking if at least 2 formants (F1, F2) are found
            F1, F2 = formants[:2]

            # Vowel Formant ranges (in Hz) for adult Male & Female
            # Male:   F1 (180-800 Hz),  F2 (800-2400 Hz)
            # Female: F1 (300-850 Hz),  F2 (800-2550 Hz)
            # These ranges are approximations and can vary based on language, culture etc.
            # We consider a window to likely contain vowel if F1 and F2 fall in these ranges

            if 180 <= F1 <= 850 and 800 <= F2 <= 2550:
                min_f1 = min(min_f1, F1)
                max_f1 = max(max_f1, F1)
                min_f2 = min(min_f2, F2)
                max_f2 = max(max_f2, F2)

    return (min_f1, max_f1), (min_f2, max_f2)


def extract_vowel_space_segments(audio_path, window_length=1.0, hop_length=0.5):
    y, sr = librosa.load(audio_path)

    # Converting window length and hop length from seconds to samples
    window_length_samples = int(window_length * sr)
    hop_length_samples = int(hop_length * sr)

    num_samples = len(y)
    num_segments = int((num_samples - window_length_samples) / hop_length_samples) + 1

    vowel_spaces = []

    for i in range(num_segments):
        start_sample = i * hop_length_samples
        end_sample = start_sample + window_length_samples

        if end_sample > num_samples:  # If end_sample exceeds audio length
            end_sample = num_samples  # Set to end of audio
            start_sample = num_samples - window_length_samples  # Set start to meet window length requirement

        segment = y[start_sample:end_sample]

        (min_f1, max_f1), (min_f2, max_f2) = calculate_vowel_space(segment, sr, window_length, hop_length)

        # Appending the min and max values of F1 and F2 formants to capture vowel space features
        vowel_spaces.append(np.array([min_f1, max_f1, min_f2, max_f2]))

    return vowel_spaces
