import tensorflow as tf
from tensorflow import expand_dims, sqrt, reduce_sum, square

def euclidean_distance(prototypes, embeddings):
    """Computes the distance of each embedding to each prototype.

    Args:
        prototypes (numpy.array): a (nb_prototypes, embedding_size)-shaped array.
        embeddings (numpy.array): a (nb_embeddings, embedding_size)-shaped array.

    Returns:
        A (nb_prototypes, nb_embeddings)-shaped array containing the euclidean distance of each embedding to each
        prototype.
    """

    expanded_prototypes = expand_dims(prototypes, 1)
    return sqrt(reduce_sum(
        square(expanded_prototypes - embeddings),
        2
    ))


def create_imageNetCNN(input_shape, nb_hidden_layers=4, nb_filters=64):
    """Creates a Keras Sequential Model as described in Matching Nets paper (Vinyals et al., 2016).

    Args:
        input_shape (int tuple): the shape of inputs passed to the model, usually (im_width, im_height, nb_channel).
        nb_hidden_layers (int): the number of convolutional blocks (conv2D, batchNorm, ReLU).
        nb_filters (int): the number of filters output by each Conv2D layer.
    """

    layers = [
        tf.keras.layers.Input(input_shape)
    ]
    for i in range(nb_hidden_layers):
        layers.extend([
            tf.keras.layers.Conv2D(nb_filters, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        ])
    layers.append(tf.keras.layers.Flatten())
    return tf.keras.models.Sequential(layers)
