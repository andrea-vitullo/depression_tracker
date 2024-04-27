import tensorflow.keras as keras
from keras.src.utils import plot_model
from keras import layers, regularizers
from keras.layers import Input, Dense, Activation, MaxPooling1D, Conv1D, Dropout, BatchNormalization, Flatten, GRU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

import my_config
from my_config import FEATURE_SHAPES, EPOCHS
from utils import utils, plots
from model import f1_metric
from data_management.data_loader import DataLoader


train_files = {
    'chroma': '/Users/andreavitullo/Desktop/Python/final_project/processed_audio_features/train_chroma.h5'
}
dev_files = {
    'chroma': '/Users/andreavitullo/Desktop/Python/final_project/processed_audio_features/dev_chroma.h5'
}
test_files = {
    'chroma': '/Users/andreavitullo/Desktop/Python/final_project/processed_audio_features/test_chroma.h5'
}


# Create DataLoader and DataGenerator with the feature shapes
data_loader = DataLoader(train_files, dev_files, test_files, FEATURE_SHAPES)

# Computes class weights for class imbalance cases
# all_labels = data_loader.get_all_labels()
# class_weights = utils.calculate_class_weights(all_labels)


chroma_input = Input(shape=FEATURE_SHAPES["chroma"][0], name='chroma')


# CHROMA LAYERS ####################################################################################################
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


full_connection = Dense(units=512, activation='relu')(chroma_net)

output_layer = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(0.1))(full_connection)

model = Model(inputs=chroma_input, outputs=output_layer)


plot_model(model, to_file='chroma_net_plot.png', show_shapes=True, show_layer_names=True)


# Define the ModelCheckpoint callback
checkpoint_filepath = 'best_chroma_model.keras'

checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=0  # Verbosity mode. 1 = progress bar
)


opt = keras.optimizers.legacy.Adam(learning_rate=my_config.INITIAL_LEARNING_RATE)


model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['binary_accuracy',
                       keras.metrics.Precision(),
                       keras.metrics.Recall(),
                       f1_metric.F1Metric(),
                       keras.metrics.AUC()])

model.summary()


callbacks = [
    EarlyStopping(monitor="val_loss", patience=50, verbose=1),
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


plots.model_plot_history(history)
