import unittest

from tensorflow_fewshot.models.utils import create_imageNetCNN
import tensorflow_fewshot.models.prototypical_network as pn
from numpy.random import normal
import tensorflow as tf
import numpy as np


class TestProtonet(unittest.TestCase):

    def setUp(self):
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2, 2, 1)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.UpSampling2D(size=(8, 8)),
            tf.keras.layers.Flatten()
        ])

        self.encoder(normal(size=(2, 2, 2, 1)))

    def test_prototypes_creation_when_calling_fit(self):
        # Given
        # Mock encoder that outputs 64x the max of a 2x2 input
        batch = np.array([
            [[4, 4],
             [4, 4]],
            [[2, 2],
             [2, 2]]
        ])[:, :, :, None]
        labels = np.array([4, 4])

        model = pn.PrototypicalNetwork(self.encoder)

        # When
        model.fit(batch, labels)

        # Then
        assert model.prototypes.shape == (1, 64)
        assert np.all(model.prototypes == 3)

    def test_predict_right_labels(self):
        # Given
        # Mock encoder that outputs 64x the max of a 2x2 input
        train_batch = np.array([
            [[4, 4],
             [4, 4]],
            [[2, 2],
             [2, 2]]
        ])[:, :, :, None]
        labels = np.array([1, 4])

        model = pn.PrototypicalNetwork(self.encoder)
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

    def test_takes_keras_model_as_input(self):
        # When
        model = pn.PrototypicalNetwork(self.encoder)

        # Then
        assert model is not None

    def test_model_doesnt_break_on_full_use_cycle(self):
        # Given
        encoder = create_imageNetCNN(input_shape=(28, 28, 1))
        meta_train_X = np.ones((2, 28, 28, 1))
        meta_train_Y = np.array((1, 2))
        train_X = np.zeros((2, 28, 28, 1))
        train_Y = np.array((2, 1))

        model = pn.PrototypicalNetwork(encoder=encoder)

        # When
        model.meta_train(meta_train_X, meta_train_Y, n_episode=2, n_way=2)
        model.fit(train_X, train_Y)
        preds = model.predict(normal(size=(3, 28, 28, 1)))

        # Then
        assert preds.shape == (3,)

    def test_model_doesnt_break_on_full_use_cycle_with_custom_model(self):
        # Given
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input((2, 2, 1)),
            tf.keras.layers.Conv2D(64, (2, 2)),
            tf.keras.layers.Flatten()
        ])
        model = pn.PrototypicalNetwork(encoder)
        meta_train_X = np.ones((2, 2, 2, 1))
        meta_train_Y = np.array((1, 2))
        train_X = np.zeros((2, 2, 2, 1))
        train_Y = np.array((2, 1))

        # When
        model.meta_train(meta_train_X, meta_train_Y, n_episode=2, n_way=2)
        model.fit(train_X, train_Y)
        preds = model.predict(normal(size=(3, 2, 2, 1)))

        # Then
        assert preds.shape == (3,)

    def test_model_raises_error_when_called_with_invalid_encoder(self):
        with self.assertRaises(TypeError):
            model = pn.PrototypicalNetwork("test")

    def test_model_raises_error_if_model_is_not_built(self):
        # Given
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(5)
        ])

        # Then
        with self.assertRaises(ValueError):
            pn.PrototypicalNetwork(encoder=encoder)
