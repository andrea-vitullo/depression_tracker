import tensorflow.keras as keras
from keras import layers, regularizers
from keras.layers import Input, LSTM, Dense, Activation, MaxPooling1D, Conv1D, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
import logging


import my_config
from my_config import LOGMEL_SHAPE_WINDOW, EPOCHS
from utils import utils
from data_loader import DataLoader
from model_tester import ModelTester
from attention_layer import Attention, squeeze_excite_block


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_loader = DataLoader(my_config.TRAIN_H5, my_config.DEV_H5, my_config.TEST_H5)


# Input layer
audio_input = Input(shape=LOGMEL_SHAPE_WINDOW)

# Conv2D Layer
conv1 = Conv1D(filters=256, kernel_size=120, strides=1, padding='valid')(audio_input)
# conv1 = squeeze_excite_block(conv1)
# conv1 = Activation('relu')(conv1)

conv1 = Dropout(0.5)(conv1)

# conv2 = Conv1D(filters=128, kernel_size=40, strides=1, padding='valid')(conv1)
# conv2 = squeeze_excite_block(conv2)
# # conv2 = Activation('relu')(conv2)

# conv2 = Dropout(0.5)(conv2)

# MaxPooling2D Layer
max_pool1 = MaxPooling1D(pool_size=1, strides=1, padding='valid')(conv1)

reshape = layers.Reshape((-1, 1))(max_pool1)

# Add the GRU layer
gru = layers.GRU(64, return_sequences=True)(reshape)
gru2 = layers.GRU(32, return_sequences=True)(gru)
gru3 = layers.LSTM(32, return_sequences=True)(gru2)
# gru4 = layers.LSTM(128, return_sequences=True)(gru3)
gru5 = layers.LSTM(32, return_sequences=False)(gru3)


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

history = model.fit(data_loader.train_dataset, epochs=epochs, validation_data=data_loader.dev_dataset, callbacks=callbacks, verbose=1)


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


######################################################################################################################


data_loader = DataLoader(my_config.TRAIN_H5, my_config.DEV_H5, my_config.TEST_H5)
tester = ModelTester('./model/1d_model.keras', data_loader)
tester.test()
