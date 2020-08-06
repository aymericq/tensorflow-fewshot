from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Lambda
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

    def test_test_2nd_order_gradient_through_updated_model(self):
        # Given
        model1 = Sequential([
            Dense(1, use_bias=False, kernel_initializer='ones', input_shape=(1,)),
            Lambda(lambda x: x ** 2)
        ])
        x = np.array([[3]])

        # When
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                y = model1(x)
            grads = inner_tape.gradient(y, model1.trainable_variables)
            model2 = take_one_gradient_step(model1, grads)
            yp = model2(x)
        grad_of_grads = outer_tape.gradient(yp, model1.trainable_variables)

        # Then
        self.assertEqual(grad_of_grads[0], 5202)

    def test_get_weights_returns_right_weights_after_update(self):
        # Given
        model = create_2_layer_MLP()
        grads = model.get_weights()

        # When
        model2 = take_one_gradient_step(model, grads, alpha=4)
        model2_weights = model2.get_weights()  # Actually different from directly getting the kernel

        # Then
        self.assertTrue((model2_weights[0] == -3*np.ones((2, 1))).all())
        self.assertTrue((model2_weights[1] == -3*np.ones((2, 1))).all())


def create_2_layer_MLP():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(2, use_bias=False, kernel_initializer='ones'),
        tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones'),
    ])
