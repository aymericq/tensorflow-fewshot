from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, clone_model
from tensorflow.python.keras.layers import Dense, Lambda, BatchNormalization
from tensorflow_fewshot.models.fast_gradients import take_n_gradient_step


class TestGradientUtils(TestCase):

    def setUp(self):
        np.random.seed(37)
        tf.random.set_seed(37)

    def test_update_weights_creates_model_with_right_weights(self):
        # Given
        initial_model = create_2_layer_MLP()
        grads = initial_model.get_weights()

        # When
        to_be_updated_model = clone_model(initial_model)
        take_n_gradient_step(
            initial_model,
            to_be_updated_model,
            n_step=1,
            alpha=1.0,
            loss=(lambda y, p: p),
            data_x=np.array([[1]]),
            data_y=np.array([[1]])
        )
        to_be_updated_model_weights = [layer.kernel for layer in to_be_updated_model.layers if layer.trainable]

        # Then
        np.testing.assert_equal(to_be_updated_model_weights[0].numpy(), np.zeros((1, 2)))
        np.testing.assert_equal(to_be_updated_model_weights[1].numpy(), np.zeros((2, 1)))

    def test_update_weights_creates_model_with_right_weights_with_alpha_2(self):
        # Given
        initial_model = create_2_layer_MLP()

        grads = initial_model.get_weights()

        # When
        updated_model = clone_model(initial_model)
        take_n_gradient_step(
            initial_model,
            updated_model,
            n_step=1,
            alpha=4.0,
            loss=(lambda y, p: p),
            data_x=np.array([[1]]),
            data_y=np.array([[1]])
        )
        updated_model_weights = [layer.kernel for layer in updated_model.layers if layer.trainable]

        # Then
        np.testing.assert_equal(updated_model_weights[0].numpy(), -3*np.ones((1, 2)))
        np.testing.assert_equal(updated_model_weights[1].numpy(), -3*np.ones((2, 1)))

    def test_2nd_order_gradient_through_updated_model(self):
        # Given
        initial_model = Sequential([
            Dense(1, use_bias=False, kernel_initializer='ones', input_shape=(1,)),
            Lambda(lambda x: x ** 2)
        ])
        x = np.array([[3]])

        updated_model = clone_model(initial_model)

        # When
        with tf.GradientTape() as outer_tape:
            take_n_gradient_step(
                initial_model,
                updated_model,
                n_step=1,
                alpha=1.0,
                loss=(lambda y, p: p),
                data_x=x,
                data_y=x
            )
            yp = updated_model(x)
        grad_of_grads = outer_tape.gradient(yp, initial_model.trainable_variables)

        # Then
        self.assertEqual(5202, grad_of_grads[0])

    def test_gradient_tape_doesnt_crash_when_model_has_non_trainable_variables(self):
        # Given
        initial_model = Sequential([
            tf.keras.layers.Input((1,)),
            Dense(3),
            BatchNormalization(),
            Dense(7)
        ])
        initial_weights = initial_model.get_weights()
        x = np.array([[1]])

        # When
        updated_model = clone_model(initial_model)
        take_n_gradient_step(
            initial_model,
            updated_model,
            n_step=1,
            alpha=1.0,
            loss=(lambda y, p: p),
            data_x=x,
            data_y=x
        )

        # Then
        np.testing.assert_equal(initial_weights[4], updated_model.get_weights()[4])  # Moving mean
        np.testing.assert_equal(initial_weights[5], updated_model.get_weights()[5])  # Moving Variance

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
        take_n_gradient_step(model, updated_model, n_step, alpha, tf.keras.losses.mse, x, y)

        # Then
        self.assertIsNotNone(updated_model(x))


    # TODO: test on different layers and models: convnets, batchnorm, conv2d, pooling, etc.


def create_2_layer_MLP():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(2, use_bias=False, kernel_initializer='ones'),
        tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones'),
    ])
