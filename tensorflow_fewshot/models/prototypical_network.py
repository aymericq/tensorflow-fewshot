from typing import Callable, Generator

import tensorflow as tf
import numpy as np

from .utils import euclidean_distance


class PrototypicalNetwork:
    """Implements Prototypical Networks from the original paper (Snell et al., 2017).

    Args:
        encoder (tf.keras.Model): The encoder used to map the image onto the metric space. Make sure that the input
            shape is defined, otherwise the model can't infer the output shape at initialization, which is required in
            later computations.
    """

    def __init__(
            self,
            encoder
    ):
        if not isinstance(encoder, tf.keras.models.Model):
            raise TypeError("Encoder must be an instance of keras.models.Model .")
        self.encoder = encoder
        try:
            self.output_dim = self.encoder.output_shape[1]
        except AttributeError:
            raise ValueError('The encoder has to be built before instantiating the model.')

        self._label_to_train_indices = {}
        self._proto_index_to_label = None
        self.prototypes = None

    def meta_train(
            self,
            task_generator: Callable[[], Generator[tuple, None, None]],
            n_episode: int,
            n_way: int,
            ks_shots: int,
            kq_shots: int,
            optimizer='Adam',
            episode_end_callback=None
    ):
        """Trains the model on the meta-training set.

        Args:
            task_generator (callable): A callable returning a generator of few_shot tasks. Each task should be a couple
                (support_set, query_set), themselves being a tuple (data, label).
            n_episode (int): Number of episodes for meta-training.
            n_way (int): Number of ways (or classes per episode).
            ks_shots (int): Number of image per class in the support set.
            kq_shots (int): Number of image per class in the query set.
            optimizer (tf.keras.optimizer): A valid Keras optimizer for training.
            episode_end_callback (Callable): callback called at the end of each episode.
        """

        lr_schedule, optimizer = self._prepare_optimizer(n_episode, optimizer)

        for episode in range(n_episode):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                support_set, query_set = task_generator().__next__()

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                distrib, support_labels, query_labels = run_episode(
                    support_set,
                    query_set,
                    n_way,
                    ks_shots,
                    kq_shots,
                    self.encoder
                )
                distrib = tf.transpose(distrib)

                # Compute the loss value for this episode.
                labels = np.array([[i] * kq_shots for i in range(n_way)]).flatten()

                loss_value = _compute_loss(distrib, query_labels, n_way)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.encoder.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

            if episode_end_callback is not None:
                args = {
                    'episode_loss': loss_value,
                    'episode_gradients': grads
                }
                episode_end_callback(**args)

    def _prepare_optimizer(self, n_episode, optimizer):
        """Compiles the model.

        Builds the optimizer and the learning schedule according to the number of episodes, then compiles the model.
        """
        if optimizer == 'Adam':
            if n_episode > 2000:
                # boundaries == [2000, 4000, ..., nb_episodes]
                boundaries = [2000 * i for i in range(1, n_episode // 2000 + 1)]
                # values == [1e-3, .5e-3, 1e-4, ...]
                values = [1e-3 / 2 ** i for i in range(n_episode // 2000 + 1)]
                lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries, values)
            else:
                lr_schedule = 1e-3
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            raise NotImplementedError

        self.encoder.compile(optimizer)
        return lr_schedule, optimizer

    def fit(self, train_x, train_y):
        """Fits the model to the data.

        Computes the prototype of each class and the internal mapping between prototype index
        and label.

        Args:
            train_x (numpy.array): An array containing training data, of shape [nb_samples, [model_input_shape]].
            train_y (numpy.array): The corresponding labels, a 1-dimensional array of integers.
        """
        n_labels = len(np.unique(train_y))
        prototypes = np.zeros((n_labels, self.output_dim)).astype(np.float32)

        self._proto_index_to_label = np.zeros((n_labels,)).astype(np.int32)

        self._label_to_train_indices = {
            ind: np.argwhere(train_y.flatten() == ind).flatten()
            for ind in np.unique(train_y)
        }

        for i, label in enumerate(np.unique(train_y)):
            prototypes[i, :] = tf.reduce_mean(
                self.encoder(train_x[self._label_to_train_indices[label], :, :, :]),
                axis=0
            )
            self._proto_index_to_label[i] = label

        self.prototypes = prototypes

    def predict(self, x):
        """Makes predictions and return the inferred labels.

        Args:
            x (numpy.array): An array containing the data to label, of shape [nb_samples, [model_input_shape]].

        Returns:
            preds (numpy.array): The predicted label for each data point, a 1-dimensional array of integers.
        """
        dists = euclidean_distance(self.prototypes, self.encoder(x).numpy())
        return self._proto_index_to_label[tf.argmin(dists, axis=0).numpy()]


def run_episode(support_set, query_set, n_way, ks_shots, kq_shots, encoder):
    """ Computes softmax of distances for one sampled episode.

    Given a set X of input images, their corresponding labels y, episode
    parameters, N-way, K-support shots, K-query shots and an encoder:
    Sample a support set and a query set according to the episode's parameters
    then computes the prototypes from the support set, and return for each image
    of the query set, the negative log-softmax of the the (negative) distance to
    each prototype.

    Args:
        support_set (tuple): couple x_support, y_support, data and corresponding labels
        query_set (tuple): couple x_query, y_query, data and corresponding labels
        n_way (int): Number of ways (or classes per episode).
        ks_shots (int): Number of image per class in the support set.
        kq_shots (int): Number of image per class in the query set.
        encoder (tf.keras.Model): The encoder used to map the image onto the metric space.

    Returns:
        dist (numpy.array): the distance of each query datapoint to each prototype.
        support_labels (numpy.array): the labels corresponding to the reordering of the support data forwarded through
            the encoder.
        query_labels (numpy.array): the labels corresponding to the reordering of the query data forwarded through
            the encoder.
    """

    x_support, y_support = support_set
    x_query, y_query = query_set

    # Compute sorted indices of labels in order to group embeddings by label and compute their mean accordingly
    i_sorted_support = np.argsort(y_support)
    i_sorted_query = np.argsort(y_query)

    # Forward support into encoder
    support_embeddings = tf.reshape(
        encoder(x_support[i_sorted_support, :, :, :]),
        (n_way, ks_shots, -1)
    )

    # Compute prototypes as mean of each class
    prototypes = tf.reduce_mean(support_embeddings, axis=1)

    # Forward pass on query set
    query_embeddings = tf.reshape(
        encoder(x_query[i_sorted_query, :, :, :]),
        (n_way * kq_shots, -1)
    )

    # Compute distances, log of opposite softmax
    distances = euclidean_distance(prototypes, query_embeddings)

    return tf.math.softmax(-distances, axis=0), y_support[i_sorted_support], y_query[i_sorted_query]


def _compute_loss(distrib, labels, n_way):
    y_true = tf.keras.utils.to_categorical(labels, num_classes=n_way)
    loss_value = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true, distrib))
    return loss_value
