from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, clone_model
from tensorflow.python.keras.layers import Dense, Lambda, BatchNormalization
from tensorflow_fewshot.models.gradient_utils import take_one_gradient_step, take_gradient_step


class TestGradientUtils(TestCase):

    def test_update_weights_creates_model_with_right_weights(self):
        # Given
        initial_model = create_2_layer_MLP()
        grads = initial_model.get_weights()

        # When
        to_be_updated_model = clone_model(initial_model)
        take_one_gradient_step(initial_model, to_be_updated_model, grads)
        to_be_updated_model_weights = [layer.kernel for layer in to_be_updated_model.layers if layer.trainable]

        # Then
        np.testing.assert_equal(to_be_updated_model_weights[0].numpy(), np.zeros((1, 2)))
        np.testing.assert_equal(to_be_updated_model_weights[1].numpy(), np.zeros((2, 1)))

    def test_update_weights_creates_model_with_right_weights_with_alpha_2(self):
        # Given
        model1 = create_2_layer_MLP()

        grads = model1.get_weights()

        # When
        model2 = clone_model(model1)
        take_one_gradient_step(model1, model2, grads, alpha=4)
        model2_weights = [layer.kernel for layer in model2.layers if layer.trainable]

        # Then
        np.testing.assert_equal(model2_weights[0].numpy(), -3*np.ones((1, 2)))
        np.testing.assert_equal(model2_weights[1].numpy(), -3*np.ones((2, 1)))

    def test_2nd_order_gradient_through_updated_model(self):
        # Given
        model1 = Sequential([
            Dense(1, use_bias=False, kernel_initializer='ones', input_shape=(1,)),
            Lambda(lambda x: x ** 2)
        ])
        x = np.array([[3]])

        model2 = clone_model(model1)

        # When
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                y = model1(x)
            grads = inner_tape.gradient(y, model1.variables, unconnected_gradients='zero')
            take_one_gradient_step(model1, model2, grads)
            yp = model2(x)
        grad_of_grads = outer_tape.gradient(yp, model1.trainable_variables)

        # Then
        self.assertEqual(5202, grad_of_grads[0])

    def test_get_weights_returns_right_weights_after_update(self):
        # Given
        model = create_2_layer_MLP()
        grads = model.get_weights()

        # When
        model2 = clone_model(model)
        take_one_gradient_step(model, model2, grads, alpha=4)
        model2_weights = model2.get_weights()  # Actually different from directly getting the kernel

        # Then
        self.assertTrue((model2_weights[0] == -3*np.ones((2, 1))).all())
        self.assertTrue((model2_weights[1] == -3*np.ones((2, 1))).all())

    def test_gradient_tape_doesnt_crash_when_model_has_non_trainable_variables(self):
        # Given
        model = Sequential([
            tf.keras.layers.Input((1,)),
            Dense(3),
            BatchNormalization(),
            Dense(7)
        ])
        initial_weights = model.get_weights()
        x = np.array([[1]])

        # When
        with tf.GradientTape() as tape:
            preds = model(x)
        grads = tape.gradient(preds, model.variables, unconnected_gradients='zero')

        updated_model = clone_model(model)
        take_one_gradient_step(model, updated_model, grads, alpha=1.0)

        # Then
        np.testing.assert_equal(grads[4], np.zeros(initial_weights[2].shape))  # Moving mean
        np.testing.assert_equal(grads[5], np.zeros(initial_weights[3].shape))  # Moving Variance
        np.testing.assert_equal(initial_weights[4], updated_model.get_weights()[4])
        np.testing.assert_equal(initial_weights[5], updated_model.get_weights()[5])

    def test_take_5_gradient_steps(self):
        # Given
        model = Sequential([
            tf.keras.layers.Input((1,)),
            Dense(1, use_bias=False, kernel_initializer='ones'),
        ])
        updated_model = clone_model(model)
        x = np.array([[1]])
        y = np.array([[4]])

        # When
        n_step = 5
        alpha = 1.0
        take_gradient_step(model, updated_model, n_step, alpha, tf.keras.losses.mse, x, y, unconnected_gradients='none')

        # Then
        self.assertIsNotNone(updated_model(x))


    # TODO: test on different layers and models: convnets, batchnorm, conv2d, pooling, etc.


def create_2_layer_MLP():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(2, use_bias=False, kernel_initializer='ones'),
        tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones'),
    ])
