import tensorflow as tf
from tensorflow import expand_dims, sqrt, reduce_sum, square

def euclidean_distance(prototypes, embeddings):
    """Compute the distance of each embedding to each prototype."""

    expanded_prototypes = expand_dims(prototypes, 1)
    return sqrt(reduce_sum(
        square(expanded_prototypes - embeddings),
        2
    ))


def create_imageNetCNN(nb_hidden_layers=4, nb_filters=64):
    """Creates a Keras Sequential Model as described in Matching Nets paper (Vinyals et al., 2016)."""

    layers = []
    for i in range(nb_hidden_layers):
        layers.extend([
            tf.keras.layers.Conv2D(nb_filters, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        ])
    layers.append(tf.keras.layers.Flatten())
    return tf.keras.models.Sequential(layers)