from unittest import TestCase

from tensorflow_fewshot.models.maml import MAML
import tensorflow as tf
import numpy as np


def create_2l_perceptron():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(2, use_bias=False),
        tf.keras.layers.Dense(1, use_bias=False),
    ])
    weights = [
        np.ones((1, 2)),
        np.ones((2, 1)),
    ]
    model.set_weights(weights)

    return model


class MAMLTest(TestCase):

    def test_instantiate_MAML(self):
        # Given
        model = tf.keras.models.Sequential()
        loss = tf.keras.losses.MSE

        # When
        maml = MAML(model, loss)

        # Then
        self.assertIsNotNone(maml)


    def test_fit_on_two_layer_perceptron_is_correct(self):
        # Given
        model = create_2l_perceptron()

        maml = MAML(model, loss=lambda y, p: p)

        data_x = np.ones((1,))
        data_y = np.zeros((1, 1))

        expected_weights = [
            np.zeros((1, 2)),
            np.zeros((2, 1)),
        ]

        # When
        maml.fit(data_x, data_y)
        weight_set = maml.model.get_weights()

        # Then
        for i_weight, weights in enumerate(weight_set):
            self.assertTrue(np.all(weights == expected_weights[i_weight]))


    def test_fit_on_two_layer_perceptron_is_correct_when_passed_a_value_for_alpha(self):
        # Given
        model = create_2l_perceptron()

        maml = MAML(model, loss=lambda y, p: p)

        data_x = np.ones((1,))
        data_y = np.zeros((1, 1))

        expected_weights = [
            0.5 * np.ones((1, 2)),
            0.5 * np.ones((2, 1)),
        ]

        # When
        maml.fit(data_x, data_y, alpha=0.5)
        weight_set = maml.model.get_weights()

        # Then
        for i_weight, weights in enumerate(weight_set):
            self.assertTrue(np.all(weights == expected_weights[i_weight]))


    def test_fit_on_two_layer_perceptron_is_correct_when_used_with_a_loss(self):
        # Given
        model = create_2l_perceptron()

        maml = MAML(model, loss=tf.keras.losses.MSE)

        data_x = np.ones((1,))
        data_y = np.zeros((1, 1))

        expected_weights = [
            -3*np.ones((1, 2)),
            -3*np.ones((2, 1)),
        ]

        # When
        maml.fit(data_x, data_y)
        weight_set = maml.model.get_weights()

        # Then
        for i_weight, weights in enumerate(weight_set):
            self.assertTrue(np.all(weights == expected_weights[i_weight]))


    def test_fit_on_two_layer_perceptron_is_correct_when_called_on_size_2_batch(self):
        # Given
        model = create_2l_perceptron()

        maml = MAML(model, loss=lambda y, p: p)

        data_x = np.ones((2,))
        data_y = np.zeros((2, 1))

        expected_weights = [
            -np.ones((1, 2)),
            -np.ones((2, 1)),
        ]

        # When
        maml.fit(data_x, data_y)
        weight_set = maml.model.get_weights()

        # Then
        for i_weight, weights in enumerate(weight_set):
            self.assertTrue(np.all(weights == expected_weights[i_weight]))
