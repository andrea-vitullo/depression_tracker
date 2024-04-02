import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, regularizers
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Activation, MaxPooling2D, Conv2D, Flatten, GlobalAveragePooling2D, TimeDistributed, Reshape, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging


import my_config
from my_config import LOGMEL_SHAPE_WINDOW, EPOCHS, BATCH_SIZE
from utils import utils
from data_generator import DataGenerator


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
audio_input = Input(shape=(*LOGMEL_SHAPE_WINDOW, 1))

# Conv2D Layer
conv1 = Conv2D(filters=256, kernel_size=(40, 3), strides=(1, 1), padding='same')(audio_input)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)

conv2 = Conv2D(filters=128, kernel_size=(40, 3), strides=(1, 1), padding='same')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)

conv2 = Dropout(0.3)(conv2)


# MaxPooling2D Layer
max_pool1 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid')(conv2)


# Preparing for LSTM
# flattened = Flatten()(max_pool1)


# reshaped = layers.Reshape((13, -1))(max_pool1)

# Assuming max_pool1 has shape (batch_size, height, width, channels)
batch_size, height, width, channels = K.int_shape(max_pool1)
reshaped = layers.Reshape((height, width * channels))(max_pool1)


# LSTM layers
lstm_layer_1 = LSTM(128, return_sequences=True)(reshaped)  # Single LSTM layer
lstm_layer_2 = LSTM(128, return_sequences=True)(lstm_layer_1)
lstm_layer_3 = LSTM(128, return_sequences=False)(lstm_layer_2)


output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(lstm_layer_3)

model = Model(inputs=audio_input, outputs=output)


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)

# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# Model summary
model.summary()


callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
    EarlyStopping(monitor="val_loss", patience=10, verbose=1),
    LearningRateScheduler(utils.lr_scheduler)
]

epochs = EPOCHS

# history = model.fit(train_dataset, epochs=epochs, validation_data=dev_dataset, callbacks=callbacks, verbose=1)


######################################################################################################################

# Save the entire model after training (Optional)
model.save('./model/model.keras')

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


# OLD MODEL   #########################################################################################################


# # Conv2D layer
# conv1_audio = Conv2D(filters=64, kernel_size=(40, 3), strides=(1, 1), padding="valid")(audio_input)
# conv1_audio = layers.BatchNormalization()(conv1_audio)
# conv1_audio = layers.Dropout(0.7)(conv1_audio)
#
# conv2_audio = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(audio_input)
# conv2_audio = layers.BatchNormalization()(conv2_audio)
# conv2_audio = layers.Dropout(0.5)(conv2_audio)
#
# conv3_audio = Reshape((-1, 32))(conv2_audio)
#
# # LSTM layers
# lstm_layer_1 = LSTM(128, return_sequences=True, dropout=0.5)(conv3_audio)
# # lstm_layer_2 = LSTM(128, return_sequences=True, dropout=0.4)(lstm_layer_1)
# lstm_layer_3 = LSTM(128, return_sequences=False, dropout=0.4)(lstm_layer_1)
#
#
# # Binary Output Layer
# hybrid_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(lstm_layer_3)
#
#
# hybrid_model = Model(inputs=audio_input, outputs=hybrid_output)
#
#
# plot_model(hybrid_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#
#
# epochs = EPOCHS
#
#
# opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)
#
#
# hybrid_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#
#
# callbacks = [
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
#     EarlyStopping(monitor="val_loss", patience=10, verbose=1),
#     LearningRateScheduler(utils.lr_scheduler)
# ]
#
#
# history = hybrid_model.fit(train_dataset,
#                            epochs=epochs,
#                            validation_data=dev_dataset,
#                            callbacks=callbacks,
#                            verbose=1)
