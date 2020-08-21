import tracemalloc
from unittest import TestCase
from unittest.mock import patch, MagicMock

from tensorflow_fewshot.models.maml import MAML
import tensorflow as tf
import numpy as np


def create_2l_perceptron():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(2, kernel_initializer='ones', use_bias=False),
        tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
    ])

    return model


def create_squared_perceptron():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
        tf.keras.layers.Lambda(lambda x: x ** 2),
    ])

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
        eval_model = maml.fit(data_x, data_y)
        weight_set = [layer.kernel for layer in eval_model.layers]

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
        eval_model = maml.fit(data_x, data_y, alpha=0.5)
        weight_set = [layer.kernel for layer in eval_model.layers]

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
            -3 * np.ones((1, 2)),
            -3 * np.ones((2, 1)),
        ]

        # When
        eval_model = maml.fit(data_x, data_y)
        weight_set = [layer.kernel for layer in eval_model.layers]

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
        eval_model = maml.fit(data_x, data_y)
        weight_set = [layer.kernel for layer in eval_model.layers]

        # Then
        for i_weight, weights in enumerate(weight_set):
            self.assertTrue(np.all(weights == expected_weights[i_weight]))

    def test_meta_learn_produces_right_model_after_1_step(self):
        # See calculus derivations in theta.py
        # Given
        model = create_squared_perceptron()
        maml = MAML(model, loss=tf.keras.losses.MSE)

        meta_train_x = np.array([
            [1],
            [2],
            [3],
        ], dtype=np.float32)
        meta_train_y = np.array([1, 2, 3])

        eval_x = np.array([
            [1],
            [2],
            [3],
        ], dtype=np.float32) * 1e-10
        eval_y = (1 - 2779171867128) ** 2 * eval_x ** 2

        def task_generator():
            for i in range(3):
                support_set = meta_train_x[i, :], meta_train_y[i]
                query_set = meta_train_x[i, :], meta_train_y[i]
                yield support_set, query_set

        # When
        optimizer = tf.keras.optimizers.SGD(1.0)
        maml.meta_train(task_generator, n_episode=1, alpha=1.0, optimizer=optimizer)
        preds = maml.model(eval_x)

        # Then
        self.assertTrue(np.all(np.abs((preds - eval_y)) < 1e-1))

    def test_task_generator_is_called_n_episode_times(self):
        # Given
        model = create_squared_perceptron()
        maml = MAML(model, loss=tf.keras.losses.MSE)
        meta_train_x = np.array([
            [1],
            [2],
            [3],
        ], dtype=np.float32)
        meta_train_y = np.array([1, -1, 7])

        def task_generator():
            for i in range(3):
                support_set = meta_train_x[i, :], meta_train_y[i]
                query_set = meta_train_x[i, :], meta_train_y[i]
                yield support_set, query_set

        # When
        magic_task_generator = MagicMock(return_value=task_generator())
        maml.meta_train(magic_task_generator, n_episode=7)

        # Then
        self.assertEqual(magic_task_generator.call_count, 7)

    def test_meta_train_doesnt_crash_with_model_with_non_trainable_variables(self):
        # Given
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((1,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1)
        ])
        maml = MAML(model, loss=tf.keras.losses.MSE)
        meta_train_x = np.array([
            [1],
            [2],
            [3],
        ], dtype=np.float32)
        meta_train_y = np.array([1, 2, 3])

        def task_generator():
            for i in range(3):
                support_set = meta_train_x[i, :], meta_train_y[i]
                query_set = meta_train_x[i, :], meta_train_y[i]
                yield support_set, query_set

        # When
        maml.meta_train(task_generator, n_episode=3)

        # Then
        self.assertIsNotNone(model(np.array([[1]])))

    def test_episode_end_callback_is_called(self):
        # Given
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((1,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1)
        ])
        maml = MAML(model, loss=tf.keras.losses.MSE)
        meta_train_x = np.array([
            [1],
            [2],
            [3],
        ], dtype=np.float32)
        meta_train_y = np.array([1, 2, 3])

        def task_generator():
            for i in range(3):
                support_set = meta_train_x[i, :], meta_train_y[i]
                query_set = meta_train_x[i, :], meta_train_y[i]
                yield support_set, query_set

        # When
        mock_callback = MagicMock()
        maml.meta_train(task_generator, n_episode=3, episode_end_callback=mock_callback)

        # Then
        for args in mock_callback.call_args_list:
            self.assertTrue('episode_loss' in args[-1])

    def test_meta_train_doesnt_grow_memory_linearly_with_n_episode(self):
        # Given
        # big beefy model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((1,)),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(100),
        ])
        maml = MAML(model, loss=tf.keras.losses.MSE)
        meta_train_x = np.array([
            [1],
            [2],
            [3],
        ], dtype=np.float32)
        meta_train_y = np.array([1, 2, 3])

        def task_generator():
            for i in range(3):
                support_set = meta_train_x[i, :], meta_train_y[i]
                query_set = meta_train_x[i, :], meta_train_y[i]
                yield support_set, query_set

        # When
        tracemalloc.start()
        maml.meta_train(task_generator, n_episode=1)
        current_mem_one_ep, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        maml.meta_train(task_generator, n_episode=10)
        current_mem_ten_ep, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Then
        self.assertLess(current_mem_ten_ep, 1.2 * current_mem_one_ep)
