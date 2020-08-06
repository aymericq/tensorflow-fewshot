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
