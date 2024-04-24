import tensorflow as tf
import tensorflow.keras as keras
from keras.src.layers import GRU
from tensorflow.keras import layers, regularizers
from keras.layers import Input, Dense, Activation, MaxPooling1D, Conv1D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np

import my_config
import plots
from my_config import FEATURE_SHAPES, EPOCHS
from utils import utils, f1_metric
from data_management.data_loader import DataLoader


train_files = {
    # 'mfcc': './processed_audio_features/train_mfcc.h5',
    'chroma': './processed_audio_features/train_chroma.h5',
    # 'logmel': './processed_audio_features/train_logmel.h5',
    # 'spectrogram': './processed_audio_features/train_spectrogram.h5',
}
dev_files = {
    # 'mfcc': './processed_audio_features/dev_mfcc.h5',
    'chroma': './processed_audio_features/dev_chroma.h5',
    # 'logmel': './processed_audio_features/dev_logmel.h5',
    # 'spectrogram': './processed_audio_features/dev_spectrogram.h5',
}
test_files = {
    # 'mfcc': './processed_audio_features/test_mfcc.h5',
    'chroma': './processed_audio_features/test_chroma.h5',
    # 'logmel': './processed_audio_features/test_logmel.h5',
    # 'spectrogram': './processed_audio_features/test_spectrogram.h5',
}


def calculate_class_weights(all_labels):
    #get unique classes
    classes = np.unique(all_labels)
    # This will compute the weights for each class to be applied during training
    class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=all_labels)
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict


# Create DataLoader and DataGenerator with the feature shapes
data_loader = DataLoader(train_files, dev_files, test_files, FEATURE_SHAPES)


# all_labels = data_loader.get_all_labels()
# class_weights = calculate_class_weights(all_labels)



# mfcc_input = Input(shape=[*FEATURE_SHAPES["mfcc"][0], 1], name='mfcc')
chroma_input = Input(shape=FEATURE_SHAPES["chroma"][0], name='chroma')
# logmel_input = Input(shape=FEATURE_SHAPES["logmel"][0], name='logmel')
# spectrogram_input = Input(shape=FEATURE_SHAPES["spectrogram"][0], name='spectrogram')


# Define separate branches of the model for each input

# MFCC LAYERS ######################################################################################################

# mfcc_net = layers.GaussianNoise(stddev=0.001)(mfcc_input)
# mfcc_net = Conv1D(128, kernel_size=1, strides=1, activation=None, padding='valid')(mfcc_net)
# mfcc_net = BatchNormalization()(mfcc_net)
# mfcc_net = Activation('tanh')(mfcc_net)
# mfcc_net = MaxPooling1D(pool_size=1, strides=2)(mfcc_net)
# mfcc_net = layers.SpatialDropout1D(rate=0.5)(mfcc_net)
# mfcc_net = Conv1D(64, kernel_size=7, strides=2, activation=None, padding='valid')(mfcc_net)
# mfcc_net = BatchNormalization()(mfcc_net)
# mfcc_net = Activation('relu')(mfcc_net)
# mfcc_net = MaxPooling1D(pool_size=3, strides=2)(mfcc_net)
# mfcc_net = Dropout(0.49951113379545005)(mfcc_net)

# mfcc_net = Conv2D(32, kernel_size=(3, 3), strides=(3, 3), activation=None, padding='same')(mfcc_input)
# mfcc_net = BatchNormalization()(mfcc_net)
# mfcc_net = Activation('tanh')(mfcc_net)
# mfcc_net = MaxPooling2D(pool_size=(4, 3), strides=(1, 3), padding='same')(mfcc_net)
# mfcc_net = Dropout(0.49951113379545005)(mfcc_net)
# mfcc_net = Conv2D(32, kernel_size=(3, 3), strides=(3, 3), activation=None, padding='same')(mfcc_net)
# mfcc_net = BatchNormalization()(mfcc_net)
# mfcc_net = Activation('tanh')(mfcc_net)
# mfcc_net = MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')(mfcc_net)
# mfcc_net = Dropout(0.49951113379545005)(mfcc_net)
#
# mfcc_net = Flatten()(mfcc_net)

# mfcc_net = LSTM(256, return_sequences=True)(mfcc_input)
# mfcc_net = LSTM(256, return_sequences=True)(mfcc_net)
# mfcc_net = LSTM(256, return_sequences=True)(mfcc_net)



######################################################################################################################

chroma_net = layers.GaussianNoise(stddev=0.001)(chroma_input)
chroma_net = Conv1D(128, kernel_size=3, strides=1, activation=None, padding='valid')(chroma_net)
chroma_net = BatchNormalization()(chroma_net)
chroma_net = Activation('tanh')(chroma_net)
chroma_net = MaxPooling1D(pool_size=2, strides=2)(chroma_net)
chroma_net = layers.SpatialDropout1D(rate=0.4)(chroma_net)
chroma_net = Conv1D(64, kernel_size=7, strides=1, activation=None, padding='valid')(chroma_net)
chroma_net = BatchNormalization()(chroma_net)
chroma_net = Activation('relu')(chroma_net)
chroma_net = MaxPooling1D(pool_size=2, strides=2)(chroma_net)
chroma_net = Conv1D(64, kernel_size=1, strides=1, activation=None, padding='valid')(chroma_net)
chroma_net = BatchNormalization()(chroma_net)
chroma_net = Activation('relu')(chroma_net)
chroma_net = MaxPooling1D(pool_size=2, strides=2)(chroma_net)

chroma_net = Dropout(0.89951113379545005)(chroma_net)

chroma_net = GRU(64, return_sequences=True)(chroma_net)

chroma_net = Flatten()(chroma_net)


# LOGMEL LAYERS ######################################################################################################

# logmel_layers1 = Conv1D(filters=128, kernel_size=3, strides=1, activation='tanh', padding='valid')(logmel_input)
# logmel_spt_dropout = layers.SpatialDropout1D(rate=0.5)(logmel_layers1)
# logmel_pooling1 = MaxPooling1D(pool_size=2, strides=2)(logmel_spt_dropout)
# logmel_layers2 = Conv1D(filters=64, kernel_size=7, strides=2, activation='relu', padding='valid')(logmel_pooling1)
# logmel_pooling2 = MaxPooling1D(pool_size=2, strides=2)(logmel_layers2)
# logmel_layers3 = Conv1D(filters=64, kernel_size=3, strides=2, activation='relu', padding='valid')(logmel_pooling2)
# logmel_pooling3 = MaxPooling1D(pool_size=2, strides=2)(logmel_layers3)
# logmel_norm = BatchNormalization()(logmel_pooling3)
# logmel_dropout1 = Dropout(0.49951113379545005)(logmel_norm)
#
# flattened_logmel = Flatten()(logmel_dropout1)


# logmel_net = Conv1D(128, kernel_size=3, strides=1, activation=None, padding='valid')(logmel_input)
# logmel_net = BatchNormalization()(logmel_net)
# logmel_net = Activation('tanh')(logmel_net)
# logmel_net = MaxPooling1D(pool_size=2, strides=2)(logmel_net)
# logmel_net = layers.SpatialDropout1D(rate=0.5)(logmel_net)
# logmel_net = Conv1D(64, kernel_size=7, strides=2, activation=None, padding='valid')(logmel_net)
# logmel_net = BatchNormalization()(logmel_net)
# logmel_net = Activation('relu')(logmel_net)
# logmel_net = MaxPooling1D(pool_size=2, strides=2)(logmel_net)
# logmel_net = Conv1D(64, kernel_size=3, strides=2, activation=None, padding='valid')(logmel_net)
# logmel_net = BatchNormalization()(logmel_net)
# logmel_net = Activation('relu')(logmel_net)
# logmel_net = MaxPooling1D(pool_size=2, strides=2)(logmel_net)
# logmel_net = Dropout(0.49951113379545005)(logmel_net)
#
# logmel_net = LSTM(64, return_sequences=True)(logmel_net)
# logmel_net = LSTM(64, return_sequences=True)(logmel_net)
#
# logmel_net = Flatten()(logmel_net)


# reshaped_layer_logmel = layers.Reshape((logmel_norm.shape[1], logmel_norm.shape[2]*logmel_norm.shape[3]))(logmel_norm)

# logmel_net = LSTM(128, return_sequences=True)(logmel_net)
# logmel_net = LSTM(128, return_sequences=True)(logmel_net)
# logmel_net = LSTM(128, return_sequences=True)(logmel_net)

# logmel_reg = layers.ActivityRegularization(l1=0.05, l2=0)(logmel_layers2)
# logmel_reg = layers.GaussianNoise(stddev=0.001)(logmel_layers2)

# logmel_gru = layers.GRU(64, return_sequences=True, dropout=0.2)(logmel_pooling2)
# logmel_gru2 = layers.GRU(64, return_sequences=True, dropout=0.2)(logmel_gru)

#####################################################################################################################

# spectrogram_net = Conv1D(128, kernel_size=3, strides=1, activation=None, padding='valid')(spectrogram_input)
# spectrogram_net = BatchNormalization()(spectrogram_net)
# spectrogram_net = Activation('tanh')(spectrogram_net)
# spectrogram_net = MaxPooling1D(pool_size=2, strides=2)(spectrogram_net)
# spectrogram_net = layers.SpatialDropout1D(rate=0.5)(spectrogram_net)
# spectrogram_net = Conv1D(64, kernel_size=7, strides=2, activation=None, padding='valid')(spectrogram_net)
# spectrogram_net = BatchNormalization()(spectrogram_net)
# spectrogram_net = Activation('tanh')(spectrogram_net)
# spectrogram_net = MaxPooling1D(pool_size=2, strides=2)(spectrogram_net)
# spectrogram_net = Conv1D(64, kernel_size=3, strides=2, activation=None, padding='valid')(spectrogram_net)
# spectrogram_net = BatchNormalization()(spectrogram_net)
# spectrogram_net = Activation('relu')(spectrogram_net)
# spectrogram_net = MaxPooling1D(pool_size=2, strides=2)(spectrogram_net)
# spectrogram_net = Dropout(0.39951113379545005)(spectrogram_net)
#
# # spectrogram_net = LSTM(64, return_sequences=True)(spectrogram_net)
# # spectrogram_net = LSTM(64, return_sequences=True)(spectrogram_net)

# spectrogram_net = Flatten()(spectrogram_net)


# full_connection_mfcc = Dense(units=256, activation='relu')(flattened_mfcc)
# full_connection_logmel = Dense(units=256, activation='relu')(flattened_logmel)

# merged_layers = Concatenate()([mfcc_net, chroma_net])


# Chroma Full Connection
# full_connection = Dense(units=256, activation='relu')(chroma_net)

# Chroma Output layer
# output_layer = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.02))(full_connection)

# Chroma Full Connection
full_connection = Dense(units=512, activation='relu')(chroma_net)

# Chroma Output layer
output_layer = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(0.1))(full_connection)

# Final model
# model = Model(inputs=[mfcc_input, chroma_input, logmel_input], outputs=output_layer)
# model = Model(inputs=[mfcc_input, chroma_input], outputs=output_layer)
model = Model(inputs=chroma_input, outputs=output_layer)


# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

checkpoint_filepath = 'best_model.keras'

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_binary_accuracy',  # Monitor the validation loss
    save_best_only=True,  # Save only the best model
    mode='min',  # The criterion to determine the best model (minimize validation loss)
    verbose=0  # Verbosity mode. 1 = progress bar
)


opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)
# opt = keras.optimizers.legacy.SGD(learning_rate=0.023934193693360832)
# opt = keras.optimizers.legacy.Adam(learning_rate=0.0023934193693360832)



model.compile(optimizer=opt,
              loss='binary_crossentropy',
              # loss=keras.losses.BinaryFocalCrossentropy(
              #     apply_class_balancing=False,
              #     alpha=0.25,
              #     gamma=2,
              #     from_logits=False,
              #     label_smoothing=0,
              #     axis=-1,
              #     reduction="sum_over_batch_size",
              #     name="binary_focal_crossentropy"
              # ),
              metrics=['binary_accuracy',
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       f1_metric.F1Metric(),
                       tf.keras.metrics.AUC()])


# Model summary
model.summary()


callbacks = [
    # ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001),
    # EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    LearningRateScheduler(utils.lr_scheduler),
    checkpoint
]


history = model.fit(data_loader.train_dataset,
                    epochs=EPOCHS,
                    validation_data=data_loader.dev_dataset,
                    # class_weight=class_weights,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=True)


######################################################################################################################

# Save the entire model after training (Optional)
model.save('./model.keras')

######################################################################################################################

plots.model_plot_history(history)
