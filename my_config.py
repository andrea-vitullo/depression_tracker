# PATHS
DIRECTORY = "/Users/andreavitullo/Desktop/DATABASE_TEST"
AUDIO_TEST_DIRECTORY = "/Users/andreavitullo/Desktop/audio_test_DAIC-WOZ_DATABASE"

TEST_SPLIT = "/Users/andreavitullo/Desktop/DATABASE_TEST/full_test_split.csv"
DEV_SPLIT = "/Users/andreavitullo/Desktop/DATABASE_TEST/dev_split.csv"
TRAIN_SPLIT = "/Users/andreavitullo/Desktop/DATABASE_TEST/train_split.csv"

# AUDIO DIRECTORIES
AUDIO_FEATURES = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_files"

AUDIO_TRAIN_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_train"
AUDIO_TRAIN_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_train/non_depressed"
AUDIO_TRAIN_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_train/depressed"

AUDIO_DEV_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_dev"
AUDIO_DEV_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_dev/non_depressed"
AUDIO_DEV_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_dev/depressed"

AUDIO_TEST_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_test"
AUDIO_TEST_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_test/non_depressed"
AUDIO_TEST_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/audio_test/depressed"

# h5 DIRECTORIES
TRAIN_H5 = '/Users/andreavitullo/Desktop/Python/final_project/processed_audio_features/train_features.h5'
DEV_H5 = '/Users/andreavitullo/Desktop/Python/final_project/processed_audio_features/dev_features.h5'
TEST_H5 = '/Users/andreavitullo/Desktop/Python/final_project/processed_audio_features/test_features.h5'

# AUDIO CONSTANTS
SAMPLERATE = 16000
MAX_LENGTH = 160000

# FORMATS
FILE_FORMAT = "wav"
START_FORMAT = "AUDIO.wav"
CLEANED_FORMAT = "cleaned.wav"
FINAL_FORMAT = "Final.wav"
SPLIT_FORMAT = "_SPLIT.wav"

LABELS = [0, 1, 2, 3]

# MFCC PARAMETERS
N_MFCC = 5
MAX_LENGTH_MFCC = 4096
MFCC_HOP_LENGTH = 512
N_FTT = 1024

MFCC_SHAPE = (480, 13)

# MEL SPECTROGRAM PARAMETERS
N_MELS = 256  # it was 80
MEL_LENGTH = 240  # it was 4096
MEL_HOP_LENGTH = 512
MEL_N_FFT = 1024
FRAME_LENGTH = 4
MEL_HOP_LENGTH_WINDOW = 32

# LOGMEL_SHAPE = (MEL_LENGTH, N_MELS, 1)
LOGMEL_SHAPE = (480, N_MELS)
SPECTROGRAM_SHAPE = (120, 513)

# RAW AUDIO PARAMETERS
NSEG = 120
H = 512

RAW_SHAPE = (NSEG * H, 1)

# CHROMA PARAMETERS
CHROMA_SHAPE = (480, 12)

# FEATURE SHAPES DICTIONARY
FEATURE_SHAPES = {
    'mfcc': (MFCC_SHAPE,),
    'chroma': (CHROMA_SHAPE,),
    'logmel': (LOGMEL_SHAPE,),
    'spectrogram': (SPECTROGRAM_SHAPE,)
}


# NN PARAMETERS
EPOCHS = 200
BATCH_SIZE = 256
NUM_CLASSES = 2  # 4 for multiclass
N_CHANNELS = 1

INITIAL_LEARNING_RATE = 0.001
DECAY_FACTOR = 0.97
EPOCHS_DROP = 20


# =============================================================

# # Assuming a standard audio segment length of 3.84 seconds as per your previous implementation
# SEGMENT_LENGTH_SEC = 3.84
#
# # Sample rate
# SR = 16000
#
# # Samples per segment
# SAMPLES_PER_SEGMENT = int(SEGMENT_LENGTH_SEC * SR)
#
# # Using the window length and hop length, for 512 sample frames with 50% overlap every frame
# N_FFT = 1024
# HOP_LENGTH = int(N_FFT / 2)
#
# # For MFCCs
# N_MFCC = 13
# MFCC_SHAPE = (481, 13)
#
# # For MEL filter-bank features
# N_MELS = 40
# MEL_SHAPE = (40, 121)
#
# # Chroma features
# N_CHROMA = 12
# CHROMA_SHAPE = (12, 121)
#
# # For (log) spectrogram
# N_FTT = 1024
# SPECTROGRAM_SHAPE = (513, 121)
#
# # Finally,
# FEATURE_SHAPES = {
#     'mfcc': MFCC_SHAPE,
#     'chroma': CHROMA_SHAPE,
#     'logmel': MEL_SHAPE,
#     'spectrogram': SPECTROGRAM_SHAPE,
# }
