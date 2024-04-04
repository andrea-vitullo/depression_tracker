import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from tensorflow.keras import backend as k


class Attention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        self.W = None
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = k.dot(x, self.W)
        e = k.tanh(e)
        a = k.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return k.sum(output, axis=1)


def squeeze_excite_block(input2d):
    """
    Create a squeeze-excite block

    Args:
        input2d: input tensor

    Returns: a keras tensor
    """

    filters = input2d.shape[-1]  # channel_axis = -1 for TF
    se = GlobalAveragePooling2D()(input2d)
    se = Reshape((1, filters))(se)
    se = Dense(filters//16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if k.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([input2d, se])

    return x


def squeeze_excite_block_1d(input1d):
    """
    Create a squeeze-excite block for 1D input

    Args:
        input1d: input tensor

    Returns: a keras tensor
    """

    filters = input1d.shape[-1]  # channel_axis = -1 for TF
    se = tf.keras.layers.GlobalAveragePooling1D()(input1d)
    se = Reshape((1, filters))(se)
    se = Dense(filters//16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([input1d, se])

    return x
