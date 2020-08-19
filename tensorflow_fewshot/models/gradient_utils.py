import tensorflow as tf
from tensorflow.keras.models import Model


def take_one_gradient_step(model: Model, cloned_model: Model, grads: list, alpha: float = 1) -> Model:
    """Updates both its numerical and trainable weights without breaking the computational graph.

    Args:
        model (Model): the model on which gradients were computed and which weights are to be updated
        grads (list): a list of numpy array, in the same order that model.get_weights provides.
        alpha (float): the magnitude of the gradient step

    Returns:
        cloned_model (Model): a cloned model of `model` with updated weights.
    """

    updated_weights = model.get_weights()
    for i in range(len(updated_weights)):
        if grads[i] is not None:
            updated_weights[i] -= alpha*grads[i]
    cloned_model.set_weights(updated_weights)

    k = 0
    for j in range(len(cloned_model.layers)):
        for var in cloned_model.layers[j].variables:
            weight_name = _extract_var_name(var)
            # Check if it is a connected gradient ie a trainable var
            if grads[k] is not None:
                cloned_model.layers[j].__dict__[weight_name] = tf.subtract(model.layers[j].__dict__[weight_name], alpha * grads[k])
            k += 1


def take_gradient_step(model, updated_model, n_step, alpha, loss, data_x, data_y, unconnected_gradients='none'):
    grads = compute_grads_n_step(model, updated_model, n_step, alpha, loss, data_x, data_y)
    update_weights(alpha, grads, model, updated_model)


def compute_grads_n_step(model, updated_model, n_step, alpha, loss, data_x, data_y, unconnected_gradients='none'):
    if n_step == 1:
        with tf.GradientTape() as tape:
            preds = model(data_x)
            loss_val = loss(data_y, preds)
        grads = tape.gradient(loss_val, model.variables, unconnected_gradients=unconnected_gradients)
        return grads
    else:
        with tf.GradientTape() as tape:
            grads = compute_grads_n_step(model, updated_model, n_step-1, alpha, loss, data_x, data_y)
            update_weights(alpha, grads, model, updated_model)
            preds = updated_model(data_x)
            loss_val = loss(data_y, preds)
        grads = tape.gradient(loss_val, model.variables, unconnected_gradients=unconnected_gradients)
        return grads

def update_weights(alpha, grads, model, updated_model):
    k = 0
    for j in range(len(updated_model.layers)):
        for var in updated_model.layers[j].variables:
            weight_name = _extract_var_name(var)
            if weight_name in [_extract_var_name(var1) for var1 in model.layers[j].trainable_variables]:
                updated_model.layers[j].__dict__[weight_name] = tf.subtract(
                    updated_model.layers[j].__dict__[weight_name],
                    alpha * grads[k]
                )
            k += 1


def _extract_var_name(var):
    return var.name.split(':')[0].split('/')[-1]
