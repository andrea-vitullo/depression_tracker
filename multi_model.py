# Adapting the price auction optimization algorithm to select ideal features from speech samples is theoretically possible, but it wouldn't be the most straightforward or recommended approach.
# Feature selection in speech and audio processing is typically more straightforward and is often guided by previous research and domain knowledge. Common choices for speech/audio feature extraction include Mel-Frequency Cepstral Coefficients (MFCCs), Chroma features, spectral contrast, and tonnetz, etc.
# Moreover, if you're dealing with a high-dimensional feature space and you want to select the most informative features, you may want to consider using feature selection techniques such as mutual information, chi-squared test, ANOVA F-value, etc. for categorical target variable, or methods like correlation coefficient, Lasso regression etc. for continuous target variable.
# If you're looking for a way to optimize parameters of your machine learning model, you would typically use techniques like Grid Search or Random Search.
# Price auction optimization algorithm is more commonly used in problems like assignment problem, transportation problem etc, where we're attempting to optimize the allocation of resources. Adapting it to a feature selection from speech samples can be quite complex and might not give you better results compared to the standard feature selection techniques.
# Therefore, I would recommend sticking to traditional feature selection methods or dimensionality reduction techniques, such as Principal Component Analysis (PCA) or Linear Discriminant Analysis (LDA).
# Remember, the goal is not just to maximize the performance of your model, but also to create a model that generalizes well and can be interpreted and understood, if possible. Choosing simpler and more established methods can often help in achieving this balance.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers
from keras.layers import Input, LSTM, Dense, Activation, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Dropout, BatchNormalization, Concatenate, Flatten
from keras.utils import plot_model
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

import my_config
from my_config import FEATURE_SHAPES, EPOCHS
from utils import utils, f1_metric
from multi_data_loader import MultiDataLoader


train_files = {
    'mfcc': './processed_audio_features/train_mfcc.h5',
    'chroma': './processed_audio_features/train_chroma.h5',
    'logmel': './processed_audio_features/train_logmel.h5',
    'spectrogram': './processed_audio_features/train_spectrogram.h5',
}
dev_files = {
    'mfcc': './processed_audio_features/dev_mfcc.h5',
    'chroma': './processed_audio_features/dev_chroma.h5',
    'logmel': './processed_audio_features/dev_logmel.h5',
    'spectrogram': './processed_audio_features/dev_spectrogram.h5',
}
test_files = {
    'mfcc': './processed_audio_features/test_mfcc.h5',
    'chroma': './processed_audio_features/test_chroma.h5',
    'logmel': './processed_audio_features/test_logmel.h5',
    'spectrogram': './processed_audio_features/test_spectrogram.h5',
}

# Create DataLoader and DataGenerator with the feature shapes
data_loader = MultiDataLoader(train_files, dev_files, test_files, FEATURE_SHAPES)


mfcc_input = Input(shape=[*FEATURE_SHAPES["mfcc"][0], 1], name='mfcc')
chroma_input = Input(shape=FEATURE_SHAPES["chroma"][0], name='chroma')
logmel_input = Input(shape=[*FEATURE_SHAPES["logmel"][0], 1], name='logmel')
spectrogram_input = Input(shape=FEATURE_SHAPES["spectrogram"][0], name='spectrogram')


# Define separate branches of the model for each input

# MFCC LAYERS ######################################################################################################
# mfcc_layers1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(mfcc_input)
# mfcc_pooling1 = MaxPooling2D(pool_size=2, strides=2)(mfcc_layers1)
# mfcc_layers2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(mfcc_pooling1)
# mfcc_pooling2 = MaxPooling2D(pool_size=2, strides=2)(mfcc_layers2)
# dropout = Dropout(0.2)(mfcc_layers2)

# mfcc_gru = layers.GRU(256, return_sequences=True, dropout=0.2)(mfcc_pooling1)

# flattened_layer = Flatten()(mfcc_pooling2)
######################################################################################################################

# chroma_layers1 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid')(chroma_input)
# chroma_layers2 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid')(chroma_layers1)
# dropout = Dropout(0.2)(chroma_layers2)
# chroma_layers3 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid')(dropout)
# chroma_pooling1 = MaxPooling1D(pool_size=8, strides=8)(chroma_layers3)
# chroma_layers4 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid')(chroma_pooling1)
# dropout = Dropout(0.2)(chroma_layers4)
# chroma_layers5 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid')(dropout)
# chroma_pooling2 = MaxPooling1D(pool_size=8, strides=8)(chroma_layers5)
# chroma_gru = layers.GRU(128, return_sequences=True, dropout=0.2)(chroma_pooling2)


# LOGMEL LAYERS ######################################################################################################
logmel_layers1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(logmel_input)
logmel_pooling1 = MaxPooling2D(pool_size=2, strides=2)(logmel_layers1)
logmel_layers2 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(logmel_pooling1)
logmel_pooling2 = MaxPooling2D(pool_size=2, strides=2)(logmel_layers2)
# dropout = Dropout(0.2)(logmel_layers1)

# logmel_gru = layers.GRU(256, return_sequences=True, dropout=0.2)(logmel_pooling1)

# Flatten the output before passing it to the Dense layer
flattened_layer = Flatten()(logmel_pooling2)
######################################################################################################################

# spectrogram_layers = Conv1D(filters=16, kernel_size=60, strides=1, padding='valid')(spectrogram_input)
# spectrogram_layers2 = Conv1D(filters=32, kernel_size=40, strides=1, padding='valid')(spectrogram_layers)
# spectrogram_layers2 = layers.Dropout(0.3)(spectrogram_layers2)
# spectrogram_pooling = MaxPooling1D(pool_size=1, strides=1)(spectrogram_layers2)
# spectrogram_gru = layers.GRU(64, return_sequences=True, dropout=0.5)(spectrogram_pooling)

# merged_layers = Concatenate()([mfcc_gru, logmel_gru])  # [mfcc_gru, chroma_gru, logmel_gru, spectrogram_gru]

# Further processing
# lstm_layer = LSTM(128, return_sequences=True, dropout=0.2)(merged_layers)
# lstm_layer3 = LSTM(128, return_sequences=False, dropout=0.2)(merged_layers)



full_connection = Dense(units=256, activation='relu')(flattened_layer)


#, kernel_regularizer=regularizers.l1(0.01)
# Output layer
output_layer = Dense(1, activation='sigmoid')(full_connection)

# Final model
model = Model(inputs=[mfcc_input, chroma_input, logmel_input, spectrogram_input], outputs=output_layer)

plot_model(model, to_file='multi_model_plot.png', show_shapes=True, show_layer_names=True)

opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       f1_metric.F1Metric(),
                       tf.keras.metrics.AUC()])


# Model summary
model.summary()


callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
    EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    # LearningRateScheduler(utils.lr_scheduler)
]


history = model.fit(data_loader.train_dataset,
                    epochs=EPOCHS,
                    validation_data=data_loader.dev_dataset,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=True)


######################################################################################################################

# Save the entire model after training (Optional)
model.save('./model/multi_model.keras')

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
