import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, Input, regularizers
from keras.layers import Reshape, LSTM, Dense
from keras.src.layers import Conv2D
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import my_config
from my_config import LOGMEL_SHAPE, EPOCHS, BATCH_SIZE
from utils import utils
from features_extractors import extract_raw_audio, extract_mfcc, extract_logmel
from preprocess import EXTRACTION_FUNCTION
from data_generator import DataGenerator


train_features, train_labels = utils.load_features('./processed_audio_features/train_features.h5')
dev_features, dev_labels = utils.load_features('./processed_audio_features/dev_features.h5')
test_features, test_labels = utils.load_features('./processed_audio_features/test_features.h5')


train_generator = DataGenerator(
    './processed_audio_features/train_features.h5',
    batch_size=BATCH_SIZE,
    audio_shape=LOGMEL_SHAPE
)

dev_generator = DataGenerator(
    './processed_audio_features/dev_features.h5',
    batch_size=BATCH_SIZE,
    audio_shape=LOGMEL_SHAPE
)


def train_gen():
    for features, labels in train_generator:
        yield features, labels


# Adjust the output_signature according to your actual data shape and types
train_dataset = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(None,) + LOGMEL_SHAPE, dtype=tf.float32),  # Adjust shape
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Adjust shape for your labels
    ),
)


def dev_gen():
    for features, labels in dev_generator:
        yield features, labels


# Adjust the output_signature according to your actual data shape and types
dev_dataset = tf.data.Dataset.from_generator(
    dev_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(None,) + LOGMEL_SHAPE, dtype=tf.float32),  # Adjust shape
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Adjust shape for your labels
    ),
)


######################################################################################################################

# buffer_size = len(train_features)

# Define a generator function for your training dataset
# def train_gen():
#     for feature, label in zip(train_features, train_labels):
#         # Ensure label is a single binary scalar value, either 0 or 1
#         yield feature, np.array([label], dtype=np.float32)
#
#
# # Use the generator function to define your training dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
# train_dataset = train_dataset.cache().shuffle(buffer_size).batch(my_config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
#
# def dev_gen():
#     for feature, label in zip(dev_features, dev_labels):
#         # Ensure label is a single binary scalar value, either 0 or 1
#         yield feature, np.array([label], dtype=np.float32)
#
#
# # Use the generator function to define your development dataset
# dev_dataset = tf.data.Dataset.from_tensor_slices((dev_features, dev_labels))
# dev_dataset = dev_dataset.cache().batch(my_config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

######################################################################################################################

# Input layer
audio_input = Input(shape=LOGMEL_SHAPE)

# Conv2D layer
conv1_audio = Conv2D(filters=64, kernel_size=(40, 3), strides=(1, 1), padding="valid")(audio_input)
conv1_audio = layers.BatchNormalization()(conv1_audio)
conv1_audio = layers.Dropout(0.7)(conv1_audio)

conv2_audio = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(audio_input)
conv2_audio = layers.BatchNormalization()(conv2_audio)
conv2_audio = layers.Dropout(0.5)(conv2_audio)

conv3_audio = Reshape((-1, 32))(conv2_audio)

# LSTM layers
lstm_layer_1 = LSTM(128, return_sequences=True, dropout=0.5)(conv3_audio)
# lstm_layer_2 = LSTM(128, return_sequences=True, dropout=0.4)(lstm_layer_1)
lstm_layer_3 = LSTM(128, return_sequences=False, dropout=0.4)(lstm_layer_1)


# Binary Output Layer
hybrid_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(lstm_layer_3)


hybrid_model = Model(inputs=audio_input, outputs=hybrid_output)


plot_model(hybrid_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


epochs = EPOCHS


opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)


hybrid_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
    EarlyStopping(monitor="val_loss", patience=10, verbose=1),
    LearningRateScheduler(utils.lr_scheduler)
]


history = hybrid_model.fit(train_dataset,
                           epochs=epochs,
                           validation_data=dev_dataset,
                           callbacks=callbacks,
                           verbose=1)


######################################################################################################################

# Save the entire model after training (Optional)
hybrid_model.save('./model/my_model_3.keras')

######################################################################################################################


# Plot the training history
def plot_history(hstory):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(hstory.history["accuracy"], label="train accuracy")
    axs[0].plot(hstory.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    # create error subplot
    axs[1].plot(hstory.history["loss"], label="train error")
    axs[1].plot(hstory.history["val_loss"], label="validation error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Evaluation")

    plt.show()


plot_history(history)

######################################################################################################################


# # Load the model if not already in memory
# # hybrid_model = keras.models.load_model('./model/my_model_3.keras')
#
# # Make predictions on the test set using the generator
# predictions = hybrid_model.predict(test_generator)
# predicted_classes = np.argmax(predictions, axis=1)
#
# # Ensure the length of test_labels matches the number of predictions
# assert len(test_labels) == len(predicted_classes), "Mismatch in number of true and predicted labels."
#
# # Compute the confusion matrix
# cm = confusion_matrix(test_labels, predicted_classes)
# plt.figure(figsize=(10, 7))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=['Non-depressed', 'Depressed'],
#     yticklabels=['Non-depressed', 'Depressed']
# )
#
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()
#
# # Print classification report
# print(classification_report(test_labels, predicted_classes, target_names=['Non-depressed', 'Depressed']))
