import tensorflow.keras as keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model
from keras.models import Model
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

import my_config

SAMPLERATE = 16000

def extract_features(file_paths, labels):
    X = []
    y = []

    for file_path, label in zip(file_paths, labels):
        audio = librosa.load(file_path, sr=SAMPLERATE)[0]
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLERATE, n_fft=1024, hop_length=160, n_mfcc=70).T[:6000]
        X.append(mfccs)
        y.append(label)

    return np.array(X), np.array(y)

train_files = glob.glob(my_config.AUDIO_TRAIN_DIR_0 + '/*.wav') + glob.glob(my_config.AUDIO_TRAIN_DIR_1 + '/*.wav')
train_labels = [0]*len(glob.glob(my_config.AUDIO_TRAIN_DIR_0 + '/*.wav')) + [1]*len(glob.glob(my_config.AUDIO_TRAIN_DIR_1 + '/*.wav'))

dev_files = glob.glob(my_config.AUDIO_DEV_DIR_0 + '/*.wav') + glob.glob(my_config.AUDIO_DEV_DIR_1 + '/*.wav')
dev_labels = [0]*len(glob.glob(my_config.AUDIO_DEV_DIR_0 + '/*.wav')) + [1]*len(glob.glob(my_config.AUDIO_DEV_DIR_1 + '/*.wav'))

# Extract features
X_train, y_train = extract_features(train_files, train_labels)
X_dev, y_dev = extract_features(dev_files, dev_labels)

n_mfcc = 70
input_shape = (6000, n_mfcc)

input_layer = keras.layers.Input(input_shape)
# Create convolutional layers
Conv1 = layers.Conv1D(16, 3, padding="same")(input_layer)
Conv1 = layers.BatchNormalization()(Conv1)
Conv1 = layers.ReLU()(Conv1)

Conv2 = layers.Conv1D(32, 3, padding="same")(Conv1)
Conv2 = layers.BatchNormalization()(Conv2)
Conv2 = layers.ReLU()(Conv2)

# Create GRU layers
Gru1 = layers.GRU(32, return_sequences=True, dropout=.3)(Conv2)
Gru2 = layers.GRU(32, return_sequences=True, dropout=.3)(Gru1)

gap = layers.GlobalAveragePooling1D()(Gru2)

# Output layer
hybrid_output = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(gap)

# Construct the model
hybrid_model = Model(input_layer, hybrid_output)

# Optional: Plot the model architecture
plot_model(hybrid_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

epochs = 48
batch_size = 64

# Define optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
hybrid_model.compile(optimizer=opt,
                     loss=keras.losses.BinaryCrossentropy(),
                     metrics=['accuracy', keras.metrics.AUC()])

# Define additional callback options
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=10, min_lr=0.001),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1)
    ]

# Fit the model
history = hybrid_model.fit(X_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(X_dev, y_dev),
                           callbacks=callbacks)

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
