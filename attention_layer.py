import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from tensorflow.keras import backend as K


class Attention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return tf.keras.backend.sum(output, axis=1)


def squeeze_excite_block(input):
    """
    Create a squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor
    """

    filters = input.shape[-1] # channel_axis = -1 for TF
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters//16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([input, se])

    return x
