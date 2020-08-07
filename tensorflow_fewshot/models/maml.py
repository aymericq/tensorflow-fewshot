from collections import generator

import numpy as np
import tensorflow as tf

from .gradient_utils import take_one_gradient_step


class MAML:
    """Implements Model-Agnostic Meta-Learning (Finn et al., 2017)."""

    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

    def fit(self, data_x, data_y, alpha=1):
        with tf.GradientTape() as tape:
            preds = self.model(data_x)
            assert preds.shape == data_y.shape
            loss_value = self.loss(data_y, preds)

        grads = tape.gradient(loss_value, self.model.weights)
        return take_one_gradient_step(self.model, grads, alpha)

    def meta_train(self, task_generator: generator, n_episode: int):
        """Meta-trains the model according to MAML algorithm.

        Args:
            task_generator (generator): A generator of few_shot tasks. Each task should be a couple
                (support_set, query_set), themselves being a tuple (data, label).
            n_episode (int): the number of episodes tu run.
        """
        for i_epi in range(n_episode):
            epi_grad = [np.zeros(weight.shape) for weight in self.model.get_weights()]
            for task in task_generator:
                support_set, query_set = task
                x_support, y_support = support_set
                x_query, y_query = query_set
                with tf.GradientTape() as outer_tape:
                    with tf.GradientTape() as inner_tape:
                        y_inner = self.model(x_support)
                        loss_val = self.loss(y_support, y_inner)
                    inner_grads = inner_tape.gradient(loss_val, self.model.trainable_variables)
                    updated_model = take_one_gradient_step(self.model, inner_grads)
                    y_outer = updated_model(x_query)
                    outer_loss = self.loss(y_query, y_outer)
                outer_grads = outer_tape.gradient(outer_loss, self.model.trainable_variables)
                for i_grad, grad in enumerate(outer_grads):
                    epi_grad[i_grad] = epi_grad[i_grad] + grad
            model_weights = self.model.get_weights()
            new_weights = [None] * len(model_weights)
            for i_weight in range(len(model_weights)):
                new_weights[i_weight] = model_weights[i_weight] - 1.0*epi_grad[i_weight]
            self.model.set_weights(new_weights)
