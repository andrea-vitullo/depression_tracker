input_spectrogram = layers.Input((None, 52))

Reshape = layers.Reshape((-1, 52, 1))(input_spectrogram)

Conv1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation ='relu')(Reshape)
Conv1 = layers.BatchNormalization()(Conv1)
Conv1 = layers.ReLU()(Conv1)


Conv2 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation ='relu')(Conv1)
Conv2 = layers.BatchNormalization()(Conv2)
Conv2 = layers.ReLU()(Conv2)


Reshape = layers.Reshape((-1, Conv2.shape[-2] * Conv2.shape[-1]))(Conv2)


Gru1 = layers.GRU(units=64, return_sequences=True, dropout=.3)(Reshape)
Gru2 = layers.GRU(units=64, return_sequences=True, dropout=.3)(Gru1)

gap = keras.layers.GlobalAveragePooling1D()(Gru2)

hybrid_output = layers.Dense(units=1, activation="sigmoid")(gap)

hybrid_model = keras.Model(input_spectrogram, hybrid_output)


plot_model(hybrid_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

epochs = 500
batch_size = 1024
# import focal_loss

hybrid_callbacks = [
    keras.callbacks.ModelCheckpoint(
        "/content/drive/MyDrive/hybrid_model/weights.{epoch:02d}-{val_loss:.4f}.hdf5.h5", monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

tf.get_logger().setLevel(logging.ERROR)


opt = keras.optimizers.SGD(learning_rate=0.01)
hybrid_model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy", keras.metrics.AUC()],
)

history = hybrid_model.fit(
    X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=conv_callbacks, validation_data=(X_valid, y_valid), verbose=1)
# last epoch 12

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()


# ----------------------------------
import librosa

def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file)  # load audio file
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=52)  # extract MFCC features
    return mfcc_features.T  # transpose to align with correct dimension

# -----------------------------------

def slidingWindow(X,window_size=32, step=10):
    windowed_mels = []
    for i in range(0, len(X) - window_size, step):
        x = X[i:(i + window_size)]
        windowed_mels.append(x)
    return np.array(windowed_mels)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def feature_exctraction(path):
    mels = []
    all_files = glob.glob(path + "/*.wav")
    dir_id = int(path.split("/")[-1])
    for ten_sec_file in all_files:
        id = int(ten_sec_file.split("/")[-1].split('.')[0])

        ten_sec_audio, _ = librosa.load(ten_sec_file, sr=SAMPLR_RATE)
        mel_spectogram = librosa.feature.melspectrogram(y=ten_sec_audio, sr=SAMPLR_RATE,
                                                        n_fft=480, hop_length=160, win_length=480, window='hamming').T
        mel_spectogram = scaler.fit_transform(mel_spectogram)
        mels.append(mel_spectogram)
    mels = np.array(mels)
    print(dir_id, mels.shape)
    return mels, dir_id


def get_mels (path):
    all_mel_spectograms = {}
    all_pathes = []
    for x in os.walk(path):
      all_pathes.append(x[0])
    all_pathes = all_pathes[1:]
    for ten_sec_audio_path in all_pathes:
      mels, id = feature_exctraction(ten_sec_audio_path)
      all_mel_spectograms[id] = mels
    return all_mel_spectograms

# Assuming model is your compiled Keras model
all_mel_spectograms = get_mels(path_to_audio_files)

for id, features in all_mel_spectograms.items():
    # Consider using slidingWindow() if you want to input data in small
    windows_features = slidingWindow(features)
    for window in windows_features:
        # Add an extra dimension for Keras (batch_size, rows, cols) => (1, rows, cols)
        window_3d = np.reshape(window, (1, window.shape[0], window.shape[1]))
        # Feed MFCC features to the model
        prediction = model.predict(window_3d)
        # Do something with prediction

audio_files = get_files_for_training()  # replace with your function to get training data files
labels = get_labels_for_training()  # replace with your function to get labels corresponding to audio_files
train_data = prepare_data_for_model(audio_files)

history = hybrid_model.fit(train_data, labels, batch_size=batch_size, epochs=epochs, callbacks=hybrid_callbacks, validation_data=(X_dev, Y_dev), verbose=1)