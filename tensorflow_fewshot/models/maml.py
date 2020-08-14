from typing import Generator, Callable

import numpy as np
import tensorflow as tf

from .gradient_utils import take_one_gradient_step


class MAML:
    """Implements Model-Agnostic Meta-Learning (Finn et al., 2017)."""

    def __init__(self, model: tf.keras.models.Model, loss: tf.keras.losses.Loss):
        """

        Args:
            model (Model): the model which training will be handled by the MAML algorithm.
            loss (Loss): the loss to use when training the model.
        """
        self.eval_model = None
        self.model = model
        self.loss = loss

    def fit(self, data_x, data_y, alpha=1):
        """Fits the model to the given (possibly few-shot) data.

        It does not updates the internal model, but rather returns an updated model that is not kept in the internal
        state of the instance.

        Args:
            data_x (np.array): an array of input data.
            data_y (np.array): an array of corresponding labels (integers).
            alpha (float): the step of the gradient step used to update the model.

        Returns:
            updated_model (Model): a fitted Keras model.
        """
        with tf.GradientTape() as tape:
            preds = self.model(data_x)
            loss_value = self.loss(data_y, preds)

        grads = tape.gradient(loss_value, self.model.weights, unconnected_gradients='zero')
        if self.eval_model is None:
            self.eval_model = tf.keras.models.clone_model(self.model)
        take_one_gradient_step(self.model, self.eval_model, grads, alpha)
        return self.eval_model

    def meta_train(
            self,
            task_generator: Callable[[], Generator[tuple, None, None]],
            n_episode: int,
            alpha: float = 1e-2,
            learning_rate: float = 1e-3,
            episode_end_callback=None,
            clip_gradient=None
    ):
        """Meta-trains the model according to MAML algorithm.

        Args:
            task_generator (generator): A generator of few_shot tasks. Each task should be a couple
                (support_set, query_set), themselves being a tuple (data, label).
            n_episode (int): the number of episodes tu run.
            alpha (float): learning rate of the inner_loop
            learning_rate (float): learning rate of the outer loop
            episode_end_callback (function): a function called at the end of each episode.
            clip_gradient : gradient extremum values
        """
        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipvalue=clip_gradient)
        updated_model = tf.keras.models.clone_model(self.model)
        for i_epi in range(n_episode):
            epi_grad = [np.zeros(weight.shape) for weight in self.model.get_weights()]
            epi_loss = 0
            for support_set, query_set in task_generator():
                x_support, y_support = support_set
                x_query, y_query = query_set
                with tf.GradientTape() as outer_tape:
                    outer_loss = self._compute_task_loss(alpha, updated_model, x_query, x_support, y_query, y_support)

                outer_grads = outer_tape.gradient(outer_loss, self.model.variables, unconnected_gradients='zero')
                for i_grad, grad in enumerate(outer_grads):
                    epi_grad[i_grad] += grad
                epi_loss += outer_loss
            sgd.apply_gradients(zip(epi_grad, self.model.variables))

            if episode_end_callback is not None:
                kwargs = {
                    'episode_gradients': epi_grad,
                    'episode': i_epi,
                    'episode_loss': epi_loss,
                }
                episode_end_callback(**kwargs)

    def _compute_task_loss(self, alpha, updated_model, x_query, x_support, y_query, y_support):
        with tf.GradientTape() as inner_tape:
            y_inner = self.model(x_support)
            loss_val = self.loss(y_support, y_inner)
        inner_grads = inner_tape.gradient(loss_val, self.model.variables, unconnected_gradients='zero')
        take_one_gradient_step(self.model, updated_model, inner_grads, alpha)
        y_outer = updated_model(x_query)
        outer_loss = self.loss(y_query, y_outer)
        return outer_loss
