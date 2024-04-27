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


SAMPLERATE = 16000


# FORMATS
FILE_FORMAT = "wav"
START_FORMAT = "AUDIO.wav"
CLEANED_FORMAT = "cleaned.wav"
FINAL_FORMAT = "Final.wav"
SPLIT_FORMAT = "_SPLIT.wav"


# N_FTT
N_FTT = 1024


# MFCC PARAMETERS
N_MFCC = 13
MFCC_MELS = 256
MAX_LENGTH_MFCC = 4096
MFCC_HOP_LENGTH = 128

MFCC_SHAPE = (480, N_MFCC)


# MEL SPECTROGRAM PARAMETERS
MEL_SPEC_MELS = 256
MEL_LENGTH = 240
MEL_HOP_LENGTH = 512

LOGMEL_SHAPE = (480, MEL_SPEC_MELS)


# SPECTROGRAM PARAMETERS
SPECTROGRAM_HOP_LENGTH = 512

SPECTROGRAM_SHAPE = (120, (N_FTT // 2 + 1))


# CHROMA PARAMETERS
CHROMA_HOP_LENGTH = 128
N_CHROMA = 12

CHROMA_SHAPE = (480, N_CHROMA)


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
NUM_CLASSES = 2
N_CHANNELS = 1

INITIAL_LEARNING_RATE = 0.001
DECAY_FACTOR = 0.97
EPOCHS_DROP = 20
