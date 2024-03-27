from features_extractors import extract_raw_audio, extract_mfcc, extract_logmel


# PATHS
DIRECTORY = "/Users/andreavitullo/Desktop/DATABASE_TEST"
AUDIO_TEST_DIRECTORY = "/Users/andreavitullo/Desktop/audio_test_DAIC-WOZ_DATABASE"

TEST_SPLIT = "/Users/andreavitullo/Desktop/DATABASE_TEST/full_test_split.csv"
DEV_SPLIT = "/Users/andreavitullo/Desktop/DATABASE_TEST/dev_split.csv"
TRAIN_SPLIT = "/Users/andreavitullo/Desktop/DATABASE_TEST/train_split.csv"

# MFCCs DIRECTORIES
MFCC_FEATURES = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_features"

TRAIN_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_train"
TRAIN_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_train/non_depressed"
TRAIN_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_train/depressed"

DEV_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_dev"
DEV_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_dev/non_depressed"
DEV_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_dev/depressed"

TEST_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_test"
TEST_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_test/non_depressed"
TEST_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/mfcc_test/depressed"


# FORMANTs DIRECTORIES
FORMANT_FEATURES = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_features"

FORM_TRAIN_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_train"
FORM_TRAIN_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_train/non_depressed"
FORM_TRAIN_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_train/depressed"

FORM_DEV_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_dev"
FORM_DEV_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_dev/non_depressed"
FORM_DEV_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_dev/depressed"

FORM_TEST_DIR = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_test"
FORM_TEST_DIR_0 = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_test/non_depressed"
FORM_TEST_DIR_1 = "/Users/andreavitullo/Desktop/DATABASE_TEST/formant_test/depressed"

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

# AUDIO CONSTANTS
SAMPLERATE = 16000
MAX_LENGTH = 320000

# FORMATS
FILE_FORMAT = "wav"
START_FORMAT = "AUDIO.wav"
CLEANED_FORMAT = "cleaned.wav"
FINAL_FORMAT = "Final.wav"
SPLIT_FORMAT = "_SPLIT.wav"

# EXTRACTION FUNCTION
# [extract_raw_audio, extract_mfcc, extract_logmel] based on extraction type to perform
EXTRACTION_FUNCTION = extract_raw_audio

# MFCC PARAMETERS
N_MFCC = 5
MAX_LENGTH_MFCC = 4096
MFCC_HOP_LENGTH = 512
N_FTT = 1024

MFCC_SHAPE = (MAX_LENGTH_MFCC, N_MFCC, 1)

# MEL SPECTROGRAM PARAMETERS
N_MELS = 80
MEL_LENGTH = 4096
MEL_HOP_LENGTH = 512
MEL_N_FFT = 1024

LOGMEL_SHAPE = (MEL_LENGTH, N_MELS, 1)

# RAW AUDIO PARAMETERS
NSEG = 120
H = 512

RAW_SHAPE = (NSEG * H, 1)

# NN PARAMETERS
EPOCHS = 200
BATCH_SIZE = 128
NUM_CLASSES = 2  # 4 for multiclass
N_CHANNELS = 1

INITIAL_LEARNING_RATE = 0.001
DECAY_FACTOR = 0.9
EPOCHS_DROP = 2  # or 3


N_F0 = 1
