import tensorflow as tf


def apply_one_gradient_step(model, grads, alpha):
    curr_weights = model.get_weights()
    new_weights = []
    for i in range(len(curr_weights)):
        new_weights.append(curr_weights[i] - alpha*grads[i])
    model.set_weights(new_weights)


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
        apply_one_gradient_step(self.model, grads, alpha)
