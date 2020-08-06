from tensorflow.keras.models import clone_model, Model
import tensorflow as tf


def take_one_gradient_step(model: Model, grads: list, alpha: float = 1) -> Model:
    """Clones `model` and updates its weights without breaking the computational graph.

    Args:
        model (Model): the model on which gradients were computed and which weights are to be updated
        grads (list): a list of numpy array, in the same order that model.get_weights provides.
        alpha (float): the magnitude of the gradient step

    Returns:
        cloned_model (Model): a cloned model of `model` with updated weights.
    """
    # On crée un clone du modèle d'origine
    # Le clone est identique au model, il a des .trainable_variables qui ont les mêmes valeurs que l'original
    cloned_model = clone_model(model)

    # On enregistre le nom des variables trainable
    for layer in cloned_model.layers:
        layer.__dict__['trainable_variable_names'] = []
        for var in layer.trainable_variables:
            name = var.name.split(':')[0].split('/')[-1]
            layer.__dict__['trainable_variable_names'].append(name)

    # On va mettre à jour les poids du modèle cloné mais sans passer par .set_weights()
    # On parcourt les layers et on affecte de nouvelles valeurs aux attributs .kernel et .bias (opérations tf)
    # L'index "k" est incrémenté de façon à récupérer à chaque fois la bonne matrice à partir de la liste "grads"
    k = 0
    for j in range(len(cloned_model.layers)):

        for trainable_var_name in cloned_model.layers[j].__dict__['trainable_variable_names']:
            # Update le kernel du layer, s'il en possède un
            cloned_model.layers[j].__dict__[name] = tf.subtract(model.layers[j].__dict__[name], alpha * grads[k])

    # /!\ WARNING : Après la mise à jour, le modèle cloné n'a plus de trainable vars. À voir si c'est problématique
    return cloned_model
