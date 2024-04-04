import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, regularizers
from keras.layers import Input, LSTM, Dense, Activation, MaxPooling1D, Conv1D, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt
import logging


import my_config
from my_config import LOGMEL_SHAPE_WINDOW, EPOCHS, BATCH_SIZE
from utils import utils
from data_generator import DataGenerator
from attention_layer import Attention, squeeze_excite_block


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


train_generator = DataGenerator(
    my_config.TRAIN_H5,
    batch_size=BATCH_SIZE,
    audio_shape=LOGMEL_SHAPE_WINDOW,
    verbose=False
)

dev_generator = DataGenerator(
    my_config.DEV_H5,
    batch_size=BATCH_SIZE,
    audio_shape=LOGMEL_SHAPE_WINDOW,
    verbose=False
)


def train_gen():
    for features, labels in train_generator:
        yield features, labels


# Adjust the output_signature according to your actual data shape and types
train_dataset = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(None,) + LOGMEL_SHAPE_WINDOW, dtype=tf.float32),  # Adjust shape
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Adjust shape for your labels
    ),
)


for features, labels in train_dataset.take(1):
    print("Features shape:", features["input_1"].shape)
    print("Labels shape:", labels.shape)


def dev_gen():
    for features, labels in dev_generator:
        yield features, labels


# Adjust the output_signature according to your actual data shape and types
dev_dataset = tf.data.Dataset.from_generator(
    dev_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(None,) + LOGMEL_SHAPE_WINDOW, dtype=tf.float32),  # Adjust shape
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Adjust shape for your labels
    ),
)


# Input layer
audio_input = Input(shape=LOGMEL_SHAPE_WINDOW)

# Conv2D Layer
conv1 = Conv1D(filters=128, kernel_size=120, strides=1, padding='valid')(audio_input)
# conv1 = squeeze_excite_block(conv1)
# conv1 = Activation('relu')(conv1)

# conv2 = Conv2D(filters=128, kernel_size=(3, 10), strides=(1, 1), padding='valid')(conv1)
# conv2 = squeeze_excite_block(conv2)
# # conv2 = Activation('relu')(conv2)
#
conv1 = Dropout(0.7)(conv1)

# MaxPooling2D Layer
max_pool1 = MaxPooling1D(pool_size=1, strides=1, padding='valid')(conv1)


# Preparing for LSTM
# flattened = layers.Flatten()(max_pool1)

reshape = layers.Reshape((-1, 1))(max_pool1)

# Add the GRU layer
gru = layers.GRU(32, return_sequences=True)(reshape)
gru2 = layers.GRU(32, return_sequences=True)(gru)
gru3 = layers.LSTM(32, return_sequences=True)(gru2)
# gru4 = layers.LSTM(128, return_sequences=True)(gru3)
gru5 = layers.LSTM(32, return_sequences=False)(gru3)

# # Assuming max_pool1 has shape (batch_size, height, width, channels)
# batch_size, height, width, channels = K.int_shape(max_pool1)
# reshaped = layers.Reshape((height, width * channels))(max_pool1)


# # LSTM layers
# lstm_layer_1 = LSTM(128, return_sequences=True)(reshaped)  # Single LSTM layer
# lstm_layer_2 = LSTM(128, return_sequences=True)(lstm_layer_1)
# lstm_layer_3 = LSTM(128, return_sequences=False)(lstm_layer_2)



output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.03))(gru5)

model = Model(inputs=audio_input, outputs=output)


plot_model(model, to_file='1d_model_plot.png', show_shapes=True, show_layer_names=True)


opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# Model summary
model.summary()


callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
    EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    LearningRateScheduler(utils.lr_scheduler)
]

epochs = EPOCHS

history = model.fit(train_dataset, epochs=epochs, validation_data=dev_dataset, callbacks=callbacks, verbose=1)


######################################################################################################################

# Save the entire model after training (Optional)
model.save('./model/1d_model.keras')

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
