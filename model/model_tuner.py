import optuna
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import (Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization,
                                     Activation, GRU)
import tensorflow as tf

from my_config import FEATURE_SHAPES
from model import f1_metric
from data_management.data_loader import DataLoader


train_files = {
    'chroma': './processed_audio_features/train_chroma.h5'
}
dev_files = {
    'chroma': './processed_audio_features/dev_chroma.h5'
}
test_files = {
    'chroma': './processed_audio_features/test_chroma.h5'
}

# Create DataLoader and DataGenerator with the feature shapes
data_loader = DataLoader(train_files, dev_files, test_files, FEATURE_SHAPES)


def objective(trial):

    # Suggest values for the hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])
    gru_units = trial.suggest_categorical('gru_units', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'tanh', 'sigmoid'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

    filters_1 = trial.suggest_categorical('filters_1', [32, 64, 128])
    kernel_size_1 = trial.suggest_categorical('kernel_size_1', [3, 5, 7])
    strides_1 = trial.suggest_categorical('strides_1', [1, 2])
    activation_1 = trial.suggest_categorical('activation_1', ['relu', 'tanh'])

    filters_2 = trial.suggest_categorical('filters_2', [32, 64, 128])
    kernel_size_2 = trial.suggest_categorical('kernel_size_2', [3, 5, 7])
    strides_2 = trial.suggest_categorical('strides_2', [1, 2])
    activation_2 = trial.suggest_categorical('activation_2', ['relu', 'tanh'])

    filters_3 = trial.suggest_categorical('filters_3', [32, 64, 128])
    kernel_size_3 = trial.suggest_categorical('kernel_size_3', [3, 5, 7])
    strides_3 = trial.suggest_categorical('strides_3', [1, 2])
    activation_3 = trial.suggest_categorical('activation_3', ['relu', 'tanh'])

    # =======================================================================

    chroma_input = Input(shape=FEATURE_SHAPES["chroma"][0], name='chroma')
    chroma_net = layers.GaussianNoise(stddev=0.001)(chroma_input)
    chroma_net = Conv1D(filters=filters_1, kernel_size=kernel_size_1, strides=strides_1, activation=None, padding='valid')(chroma_net)
    chroma_net = BatchNormalization()(chroma_net)
    chroma_net = Activation(activation_1)(chroma_net)
    chroma_net = MaxPooling1D(pool_size=2, strides=2)(chroma_net)
    chroma_net = layers.SpatialDropout1D(rate=0.4)(chroma_net)
    chroma_net = Conv1D(filters=filters_2, kernel_size=kernel_size_2, strides=strides_2, activation=None, padding='valid')(chroma_net)
    chroma_net = BatchNormalization()(chroma_net)
    chroma_net = Activation(activation_2)(chroma_net)
    chroma_net = MaxPooling1D(pool_size=2, strides=2)(chroma_net)
    chroma_net = Conv1D(filters=filters_3, kernel_size=kernel_size_3, strides=strides_3, activation=None, padding='valid')(chroma_net)
    chroma_net = BatchNormalization()(chroma_net)
    chroma_net = Activation(activation_3)(chroma_net)
    chroma_net = MaxPooling1D(pool_size=2, strides=2)(chroma_net)

    chroma_net = Dropout(dropout_rate)(chroma_net)

    chroma_net = GRU(units=gru_units, return_sequences=True)(chroma_net)

    chroma_net = Flatten()(chroma_net)

    full_connection = Dense(units=dense_units, activation=activation_function)(chroma_net)
    output_layer = Dense(1, activation='sigmoid')(full_connection)

    model = Model(inputs=chroma_input, outputs=output_layer)

    # Dynamically setting the optimizer
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

    # Compilation
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           f1_metric.F1Metric(),
                           tf.keras.metrics.AUC()])

    # Training (adjust according to your data loaders)
    history = model.fit(data_loader.train_dataset, epochs=50, validation_data=data_loader.dev_dataset, verbose=0)

    # Objective: maximize validation accuracy
    best_accuracy = max(history.history['val_accuracy'])
    return best_accuracy


# Enable default logger to see the progress in the console.
optuna.logging.enable_default_handler()

# Create a study object without specifying direction
study = optuna.create_study(direction='maximize')

# Proceed with optimization
study.optimize(objective, n_trials=20)

# After optimization, you can disable logging if you prefer.
optuna.logging.disable_default_handler()

# Print the best hyperparameters
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')


# Best trial:
#   Value: 0.6733668446540833
#   Params:
#     learning_rate: 0.0007012739228112294
#     dense_units: 256
#     gru_units: 64
#     dropout_rate: 0.33199114415965136
#     activation_function: sigmoid
#     optimizer: adam
#     filters_1: 64
#     kernel_size_1: 5
#     strides_1: 2
#     activation_1: tanh
#     filters_2: 32
#     kernel_size_2: 7
#     strides_2: 2
#     activation_2: tanh
#     filters_3: 128
#     kernel_size_3: 3
#     strides_3: 2
#     activation_3: relu

