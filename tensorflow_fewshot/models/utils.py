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


def create_imageNetCNN(input_shape, nb_hidden_layers=4, nb_filters=64, use_dense_head=False, output_dim=None):
    """Creates a Keras Sequential Model as described in Matching Nets paper (Vinyals et al., 2016).

    Args:
        input_shape (int tuple): the shape of inputs passed to the model, usually (im_width, im_height, nb_channel).
        nb_hidden_layers (int): the number of convolutional blocks (conv2D, batchNorm, ReLU).
        nb_filters (int): the number of filters output by each Conv2D layer.
        use_dense_head (bool): when True, adds a dense layer on top of the encoder.
        output_dim (int): the dimension of the output when using a dense head.
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
    if use_dense_head:
        if output_dim is None:
            raise ValueError("When using a dense head, you must specify the output dimension.")
        layers.append(tf.keras.layers.Dense(output_dim))
        layers.append(tf.keras.layers.Softmax())
    return tf.keras.models.Sequential(layers)
