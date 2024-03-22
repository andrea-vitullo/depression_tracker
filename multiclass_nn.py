import tensorflow.keras as keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model, to_categorical
from keras.models import Model
import glob
import librosa
from librosa.effects import time_stretch, pitch_shift

import numpy as np
import matplotlib.pyplot as plt

import my_config

SAMPLERATE = 16000


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def extract_features(file_paths, labels, augment=False):
    X = []
    y = []

    for file_path, label in zip(file_paths, labels):
        audio, sr = librosa.load(file_path, sr=SAMPLERATE)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_fft=1024, hop_length=160, n_mfcc=70).T[:7000]
        X.append(mfccs)
        y.append(label)

        if augment and label == 0:  # Assuming '0' is the label for the non-depressed class
            # Apply different augmentations
            for _ in range(non_depressed_augmentations):  # Define how many augmented versions you want

                noise_audio = add_noise(audio)

                # Extract MFCCs from augmented audio and add to the dataset
                augmented_mfccs = librosa.feature.mfcc(y=noise_audio, sr=sr, n_fft=1024, hop_length=160, n_mfcc=70).T[:7000]
                X.append(augmented_mfccs)
                y.append(label)

        # Check if augmentation is to be applied and if the sample is from the depressed class
        if augment and label == 1:  # Assuming '1' is the label for the depressed class
            # Apply different augmentations
            for _ in range(depressed_augmentations):  # Define how many augmented versions you want

                noise_audio = add_noise(audio)

                # # Time-stretch the waveform (speed up or slow down)
                # stretch_rate = np.random.uniform(0.8, 1.2)
                # stretched_audio = time_stretch(y=audio, rate=stretch_rate)
                #
                # # Pitch-shift the waveform (up or down in semitones)
                # n_steps = np.random.randint(-1, 2)
                # shifted_audio = pitch_shift(stretched_audio, sr=sr, n_steps=n_steps)

                # Extract MFCCs from augmented audio and add to the dataset
                augmented_mfccs = librosa.feature.mfcc(y=noise_audio, sr=sr, n_fft=1024, hop_length=160, n_mfcc=70).T[:7000]
                X.append(augmented_mfccs)
                y.append(label)

    return np.array(X), np.array(y)


# Define the number of augmented versions to generate for each depressed class sample
depressed_augmentations = 2  # for example, to double the number of depressed cases
non_depressed_augmentations = 1

train_files_male_nd = glob.glob(my_config.AUDIO_TRAIN_DIR_0 + '/male/*.wav')  # non_depressed
train_files_female_nd = glob.glob(my_config.AUDIO_TRAIN_DIR_0 + '/female/*.wav')
train_files_male_d = glob.glob(my_config.AUDIO_TRAIN_DIR_1 + '/male/*.wav')  # depressed
train_files_female_d = glob.glob(my_config.AUDIO_TRAIN_DIR_1 + '/female/*.wav')

dev_files_male_nd = glob.glob(my_config.AUDIO_DEV_DIR_0 + '/male/*.wav')  # non_depressed
dev_files_female_nd = glob.glob(my_config.AUDIO_DEV_DIR_0 + '/female/*.wav')
dev_files_male_d = glob.glob(my_config.AUDIO_DEV_DIR_1 + '/male/*.wav')  # depressed
dev_files_female_d = glob.glob(my_config.AUDIO_DEV_DIR_1 + '/female/*.wav')

train_files = train_files_male_nd + train_files_female_nd + train_files_male_d + train_files_female_d
train_labels = [0]*len(train_files_male_nd) + [1]*len(train_files_female_nd) + \
               [2]*len(train_files_male_d) + [3]*len(train_files_female_d)

dev_files = dev_files_male_nd + dev_files_female_nd + dev_files_male_d + dev_files_female_d
dev_labels = [0]*len(dev_files_male_nd) + [1]*len(dev_files_female_nd) + \
             [2]*len(dev_files_male_d) + [3]*len(dev_files_female_d)

# During feature extraction, set augment=True for training set
X_train, y_train = extract_features(train_files, train_labels, augment=True)
X_dev, y_dev = extract_features(dev_files, dev_labels, augment=False)  # Typically no augmentation on validation set

# one-hot encode the labels for the neural network
y_train = to_categorical(y_train, num_classes=4)
y_dev = to_categorical(y_dev, num_classes=4)

n_mfcc = 70
input_shape = (7000, n_mfcc)

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

# Output layer for multi-class
hybrid_output = layers.Dense(4, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(gap)

# Construct the model
hybrid_model = Model(input_layer, hybrid_output)

# Optional: Plot the model architecture
plot_model(hybrid_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

epochs = 500
batch_size = 64

# Define optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)

# Compile the model for multi-class
hybrid_model.compile(optimizer=opt,
                     loss='categorical_crossentropy',  # Updated loss function for multi-class
                     metrics=['accuracy'])

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
