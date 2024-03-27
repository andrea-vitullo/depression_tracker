import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, Input, regularizers
from keras.layers import Conv1D, Reshape, LSTM, Dense
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import librosa
from librosa.effects import time_stretch, pitch_shift
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import my_config
from my_config import SAMPLERATE, MAX_LENGTH, LABELS, EPOCHS, NSEG, H
from utils import audio_utils
from utils import utils
from features_extractors import extract_raw_audio, extract_mfcc, extract_logmel
from preprocess import preprocess_and_save_features


# EXTRACTION FUNCTION
# [extract_raw_audio, extract_mfcc, extract_logmel] based on extraction type to perform
EXTRACTION_FUNCTION = extract_raw_audio


# Define directory and label mappings
AUDIO_TRAIN_DIRS = [my_config.AUDIO_TRAIN_DIR_0, my_config.AUDIO_TRAIN_DIR_0, my_config.AUDIO_TRAIN_DIR_1, my_config.AUDIO_TRAIN_DIR_1]
AUDIO_DEV_DIRS = [my_config.AUDIO_DEV_DIR_0, my_config.AUDIO_DEV_DIR_0, my_config.AUDIO_DEV_DIR_1, my_config.AUDIO_DEV_DIR_1]
AUDIO_TEST_DIRS = [my_config.AUDIO_TEST_DIR_0, my_config.AUDIO_TEST_DIR_0, my_config.AUDIO_TEST_DIR_1, my_config.AUDIO_TEST_DIR_1]

# Load the data
train_files, train_labels = utils.load_files_labels(AUDIO_TRAIN_DIRS, LABELS)
dev_files, dev_labels = utils.load_files_labels(AUDIO_DEV_DIRS, LABELS)
test_files, test_labels = utils.load_files_labels(AUDIO_TEST_DIRS, LABELS)


preprocess_and_save_features(
    train_files,
    train_labels,
    './processed_audio_features/train_features.h5',
    augment=True,
    extraction_func=EXTRACTION_FUNCTION
)

preprocess_and_save_features(
    dev_files,
    dev_labels,
    './processed_audio_features/dev_features.h5',
    augment=False,
    extraction_func=EXTRACTION_FUNCTION
)

preprocess_and_save_features(
    test_files,
    test_labels,
    './processed_audio_features/test_features.h5',
    augment=False,
    extraction_func=EXTRACTION_FUNCTION
)

train_generator, dev_generator, test_generator = utils.create_datagenerator(
    EXTRACTION_FUNCTION,
    './processed_audio_features/train_features.h5',
    './processed_audio_features/dev_features.h5',
    './processed_audio_features/test_features.h5',
    my_config.BATCH_SIZE
)


# Define a generator function for your training dataset
def train_gen():
    for item in train_generator:
        yield item

# Use the generator function to define your training dataset
train_dataset = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(my_config.BATCH_SIZE, NSEG * H, 1), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(my_config.BATCH_SIZE, my_config.NUM_CLASSES), dtype=tf.float32),
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
            "input_1": tf.TensorSpec(shape=(my_config.BATCH_SIZE, NSEG * H, 1), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(my_config.BATCH_SIZE, my_config.NUM_CLASSES), dtype=tf.float32),
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
opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)

# # Compile the model for multi-class
# hybrid_model.compile(optimizer=opt,
#                      loss='categorical_crossentropy',
#                      metrics=['accuracy'])

hybrid_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Define additional callback options
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
    EarlyStopping(monitor="val_loss", patience=10, verbose=1),
    LearningRateScheduler(utils.lr_scheduler)
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
