
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

# FORMATS
FILE_FORMAT = "wav"
START_FORMAT = "AUDIO.wav"
CLEANED_FORMAT = "cleaned.wav"
FINAL_FORMAT = "Final.wav"
SPLIT_FORMAT = "_SPLIT.wav"

# MFCC AND NN PARAMETERS
N_MFCC = 20
N_FRAMES = 200
NUM_CLASSES = 1
EPOCHS = 5

# SEQUENCE LENGTH of the sliding window for LSTM.
SEQ_LEN = 100  # 3.2 seconds

# PARAMETERS FOR SPECTRAL AUGMENTATION
SPECAUG_FREQ_MASK_PROB = 0.3
SPECAUG_TIME_MASK_PROB = 0.3
SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5
