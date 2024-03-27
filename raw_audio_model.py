import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, Input, regularizers
from keras.layers import Conv1D, Reshape, LSTM, Dense
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import glob
import librosa
from librosa.effects import time_stretch, pitch_shift
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from my_config import *
from features_extractors import extract_raw_audio
from utils import audio_utils
from data_generator import DataGenerator


def preprocess_and_save_features(file_paths, labels, output_file_path, augment=False):
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(output_file_path, 'w') as h5f:
        for i, (file_path, label) in enumerate(zip(file_paths, labels)):
            print(f"Processing file: {file_path}")
            audio, sr = librosa.load(file_path, sr=SAMPLERATE)

            if len(audio) > MAX_LENGTH:
                audio = audio[:MAX_LENGTH]
            elif len(audio) < MAX_LENGTH:
                padding = MAX_LENGTH - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')

            audio_features_padded = extract_raw_audio(audio, sr)

            # Map multi-class labels to binary labels here
            if label in [0, 1]:    # Non-depressed
                binary_label = 0
            elif label in [2, 3]:  # Depressed
                binary_label = 1

            grp = h5f.create_group(str(i))
            grp.create_dataset('audio', data=audio_features_padded, compression='gzip')
            grp.attrs['label'] = binary_label

            if augment:
                # Determine the number of augmentations based on the label
                if binary_label == 0:
                    num_augmentations = male_non_depressed_augmentations + female_non_depressed_augmentations
                elif binary_label == 1:
                    num_augmentations = male_depressed_augmentations + female_depressed_augmentations

                for aug_index in range(num_augmentations):
                    augmentation_type = random.choice(['noise'])
                    print(f"Augmentation type: {augmentation_type}")

                    if augmentation_type == 'noise':
                        augmented_audio = audio_utils.add_noise(audio)
                    # elif augmentation_type == 'stretch':
                        # stretch_rate = random.uniform(0.9, 1.1)
                        # augmented_audio = time_stretch(audio, rate=stretch_rate)
                    # elif augmentation_type == 'shift':
                    #     n_steps = random.randint(-1, 1)
                    #     augmented_audio = pitch_shift(audio, sr=sr, n_steps=n_steps)

                    augmented_mfcc_features = extract_raw_audio(augmented_audio, sr)
                    aug_grp = h5f.create_group(f"{i}_aug_{aug_index}")
                    aug_grp.create_dataset('audio', data=augmented_mfcc_features, compression='gzip')
                    aug_grp.attrs['label'] = binary_label

                    print(f"Augmented audio label: {binary_label}")



# Define the number of augmented versions to generate for each depressed class sample
male_non_depressed_augmentations = 0
female_non_depressed_augmentations = 1
male_depressed_augmentations = 2
female_depressed_augmentations = 2

train_files_male_nd = glob.glob(AUDIO_TRAIN_DIR_0 + '/male/*.wav')  # non_depressed
train_files_female_nd = glob.glob(AUDIO_TRAIN_DIR_0 + '/female/*.wav')
train_files_male_d = glob.glob(AUDIO_TRAIN_DIR_1 + '/male/*.wav')  # depressed
train_files_female_d = glob.glob(AUDIO_TRAIN_DIR_1 + '/female/*.wav')

dev_files_male_nd = glob.glob(AUDIO_DEV_DIR_0 + '/male/*.wav')  # non_depressed
dev_files_female_nd = glob.glob(AUDIO_DEV_DIR_0 + '/female/*.wav')
dev_files_male_d = glob.glob(AUDIO_DEV_DIR_1 + '/male/*.wav')  # depressed
dev_files_female_d = glob.glob(AUDIO_DEV_DIR_1 + '/female/*.wav')

train_files = train_files_male_nd + train_files_female_nd + train_files_male_d + train_files_female_d
train_labels = [0]*len(train_files_male_nd) + [1]*len(train_files_female_nd) + \
               [2]*len(train_files_male_d) + [3]*len(train_files_female_d)

dev_files = dev_files_male_nd + dev_files_female_nd + dev_files_male_d + dev_files_female_d
dev_labels = [0]*len(dev_files_male_nd) + [1]*len(dev_files_female_nd) + \
             [2]*len(dev_files_male_d) + [3]*len(dev_files_female_d)




# Check first few labels
for file, label in zip(train_files[:10], train_labels[:10]):
    print(f"File: {file}, Label: {label}")

# Check last few labels
for file, label in zip(train_files[-10:], train_labels[-10:]):
    print(f"File: {file}, Label: {label}")

assert len(train_files) == len(train_labels), "Mismatch between number of files and labels in training data."
assert len(dev_files) == len(dev_labels), "Mismatch between number of files and labels in development data."

from collections import Counter

print(Counter(train_labels))
print(Counter(dev_labels))

preprocess_and_save_features(train_files, train_labels, './processed_audio_features/train_features.h5', augment=True)
preprocess_and_save_features(dev_files, dev_labels, './processed_audio_features/dev_features.h5', augment=False)

train_generator = DataGenerator('./processed_audio_features/train_features.h5', batch_size=BATCH_SIZE)
dev_generator = DataGenerator('./processed_audio_features/dev_features.h5', batch_size=BATCH_SIZE)


def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % EPOCHS_DROP == 0:
        return lr * DECAY_FACTOR
    return lr


# Define a generator function for your training dataset
def train_gen():
    for item in train_generator:
        yield item

# Use the generator function to define your training dataset
train_dataset = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(BATCH_SIZE, NSEG * H, 1), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(BATCH_SIZE, NUM_CLASSES), dtype=tf.float32),
    ),
)

# Define a generator function for your development dataset
def dev_gen():
    for item in dev_generator:
        yield item

# Use the generator function to define your development dataset
dev_dataset = tf.data.Dataset.from_generator(
    dev_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(BATCH_SIZE, NSEG * H, 1), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(BATCH_SIZE, NUM_CLASSES), dtype=tf.float32),
    ),
)


# Input layer for raw audio
audio_input = Input(shape=(NSEG * H, 1))

# Conv1D layers
conv1_audio = Conv1D(filters=128, kernel_size=512, strides=256, padding="same")(audio_input)
conv1_audio = layers.BatchNormalization()(conv1_audio)
conv1_audio = layers.Dropout(0.6)(conv1_audio)

# conv2_audio = Conv1D(filters=64, kernel_size=3, strides=1, padding="same")(conv1_audio)
# conv2_audio = layers.BatchNormalization()(conv2_audio)
# conv2_audio = layers.Dropout(0.5)(conv2_audio)

conv3_audio = Reshape((-1, 32))(conv1_audio)

# LSTM layers
lstm_layer_1 = LSTM(128, return_sequences=True, dropout=0.5)(conv3_audio)
# lstm_layer_2 = LSTM(128, return_sequences=True, dropout=0.4)(lstm_layer_1)
lstm_layer_3 = LSTM(128, return_sequences=False, dropout=0.5)(lstm_layer_1)

# Multiclass Output Layer
# hybrid_output = layers.Dense(4, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(lstm_layer_3)

# Binary Output Layer
hybrid_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.7))(lstm_layer_3)

hybrid_model = Model(inputs=audio_input, outputs=hybrid_output)

plot_model(hybrid_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

epochs = EPOCHS

# Define optimizer
opt = keras.optimizers.legacy.Adam(learning_rate=INITIAL_LEARNING_RATE)

# # Compile the model for multi-class
# hybrid_model.compile(optimizer=opt,
#                      loss='categorical_crossentropy',
#                      metrics=['accuracy'])

hybrid_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Define additional callback options
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
    EarlyStopping(monitor="val_loss", patience=10, verbose=1),
    LearningRateScheduler(lr_scheduler)
]

# Fit the model
history = hybrid_model.fit(train_generator,
                           epochs=epochs,
                           validation_data=dev_generator,
                           callbacks=callbacks,
                           verbose=1)


# Save the entire model after training (Optional)
hybrid_model.save('./model/my_model_3.keras')

# Plot the training history
def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Evaluation")

    plt.show()


plot_history(history)


# Load test files
test_files_male_nd = glob.glob(AUDIO_TEST_DIR_0 + '/male/*.wav')
test_files_female_nd = glob.glob(AUDIO_TEST_DIR_0 + '/female/*.wav')
test_files_male_d = glob.glob(AUDIO_TEST_DIR_1 + '/male/*.wav')
test_files_female_d = glob.glob(AUDIO_TEST_DIR_1 + '/female/*.wav')


# Assuming you have a test set ready similar to train and dev sets
test_files = test_files_male_nd + test_files_female_nd + test_files_male_d + test_files_female_d
test_labels = [0]*len(test_files_male_nd) + [0]*len(test_files_female_nd) + \
              [1]*len(test_files_male_d) + [1]*len(test_files_female_d)

preprocess_and_save_features(test_files, test_labels, './processed_audio_features/test_features.h5', augment=False)

# Assuming `DataGenerator` is correctly implemented for loading test data
test_generator = DataGenerator('./processed_audio_features/test_features.h5', batch_size=BATCH_SIZE)

# Load the model if not already in memory
# hybrid_model = keras.models.load_model('./model/my_model_3.keras')

# Make predictions on the test set using the generator
predictions = hybrid_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Ensure the length of test_labels matches the number of predictions
assert len(test_labels) == len(predicted_classes), "Mismatch in number of true and predicted labels."

# Compute the confusion matrix
cm = confusion_matrix(test_labels, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=['Non-depressed', 'Depressed'],
    yticklabels=['Non-depressed', 'Depressed']
)

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(test_labels, predicted_classes, target_names=['Non-depressed', 'Depressed']))
