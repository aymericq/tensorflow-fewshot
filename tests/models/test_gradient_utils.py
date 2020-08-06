from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow_fewshot.models.gradient_utils import take_one_gradient_step


class TestGradientUtils(TestCase):

    def test_update_weights_creates_model_with_right_weights(self):
        # Given
        model1 = create_2_layer_MLP()

        grads = model1.get_weights()

        # When
        model2 = take_one_gradient_step(model1, grads)
        model2_weights = [layer.kernel for layer in model2.layers if layer.trainable]

        # Then
        self.assertTrue((model2_weights[0] == -np.zeros((2, 1))).numpy().all())
        self.assertTrue((model2_weights[1] == -np.zeros((2, 1))).numpy().all())

    def test_update_weights_creates_model_with_right_weights_with_alpha_2(self):
        # Given
        model1 = create_2_layer_MLP()

        grads = model1.get_weights()

        # When
        model2 = take_one_gradient_step(model1, grads, alpha=4)
        model2_weights = [layer.kernel for layer in model2.layers if layer.trainable]

        # Then
        self.assertTrue((model2_weights[0] == -3*np.ones((2, 1))).numpy().all())
        self.assertTrue((model2_weights[1] == -3*np.ones((2, 1))).numpy().all())

def create_2_layer_MLP():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(2, use_bias=False, kernel_initializer='ones'),
        tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones'),
    ])