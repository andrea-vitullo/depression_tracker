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

MFCC_SHAPE = (120, 13)

# MEL SPECTROGRAM PARAMETERS
N_MELS = 256 # it was 80
MEL_LENGTH = 240 # it was 4096
MEL_HOP_LENGTH = 512
MEL_N_FFT = 1024
FRAME_LENGTH = 4
MEL_HOP_LENGTH_WINDOW = 32

# LOGMEL_SHAPE = (MEL_LENGTH, N_MELS, 1)
LOGMEL_SHAPE = (120, N_MELS)
SPECTROGRAM_SHAPE = (120, 513)

# RAW AUDIO PARAMETERS
NSEG = 120
H = 512

RAW_SHAPE = (NSEG * H, 1)

# CHROMA PARAMETERS
CHROMA_SHAPE = (120, 128)

# FEATURE SHAPES DICTIONARY
FEATURE_SHAPES = {
    'mfcc': (MFCC_SHAPE,),  # Replace with your actual MFCC shape
    'chroma': (CHROMA_SHAPE,),  # Replace with your actual Chroma shape
    'logmel': (LOGMEL_SHAPE,),
    'spectrogram': (SPECTROGRAM_SHAPE,),  # Replace with your actual Spectrogram shape
}


# NN PARAMETERS
EPOCHS = 200
BATCH_SIZE = 64
NUM_CLASSES = 2  # 4 for multiclass
N_CHANNELS = 1

INITIAL_LEARNING_RATE = 0.0001
DECAY_FACTOR = 0.9
EPOCHS_DROP = 6
