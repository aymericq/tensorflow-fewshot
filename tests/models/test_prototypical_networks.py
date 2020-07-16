import tensorflow_fewshot.models.prototypical_network as pn
from numpy.random import normal
import tensorflow as tf
import numpy as np


def test_l2_norm_shape_is_10_5_when_passed_10_prototypes_10_embeddings():
    # Given
    prototypes = normal(size=(5, 64))
    query_embeddings = normal(size=(10, 64))

    # When
    distances = pn._euclidean_distance(prototypes, query_embeddings)

    # Then
    assert distances.shape == (5, 10)


def test_encoder_output_shape_is_10_64_when_passed_10_images():
    # Given
    encoder = pn._create_imageNetCNN()
    images = normal(size=(10, 28, 28, 1))

    # When
    embeddings = encoder(images)

    # Then
    assert embeddings.shape == (10, 64)


def test_prototypes_creation_when_calling_fit():
    # Given
    # Mock encoder that outputs 64x the max of a 2x2 input
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2, 2, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.UpSampling2D(size=(8, 8)),
        tf.keras.layers.Flatten()
    ])

    batch = np.array([
        [[4, 4],
         [4, 4]],
        [[2, 2],
         [2, 2]]
    ])[:, :, :, None]
    labels = np.array([4, 4])

    model = pn.PrototypicalNetwork()
    model.encoder = encoder

    # When
    model.fit(batch, labels)

    # Then
    assert model.prototypes.shape == (1, 64)
    assert np.all(model.prototypes == 3)


def test_predict_right_labels():
    # Given
    # Mock encoder that outputs 64x the max of a 2x2 input
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2, 2, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.UpSampling2D(size=(8, 8)),
        tf.keras.layers.Flatten()
    ])

    train_batch = np.array([
        [[4, 4],
         [4, 4]],
        [[2, 2],
         [2, 2]]
    ])[:, :, :, None]
    labels = np.array([1, 4])

    model = pn.PrototypicalNetwork()
    model.encoder = encoder
    model.fit(train_batch, labels)

    test_batch = np.array([
        [[1, 1],
         [1, 1]],
        [[2, 2],
         [2, 2]]
    ])[:, :, :, None]

    # When
    preds = model.predict(test_batch)

    # Then
    assert preds.shape == (2,)
    assert list(preds) == [4, 4]
