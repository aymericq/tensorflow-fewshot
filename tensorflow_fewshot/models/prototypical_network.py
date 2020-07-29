import tensorflow as tf
import numpy as np

from .utils import euclidean_distance, create_imageNetCNN


class PrototypicalNetwork:
    """Implements Prototypical Networks from the original paper (Snell et al., 2017).

    Args:
        encoder (tf.keras.Model): The encoder used to map the image onto the metric space.
    """

    def __init__(
            self,
            encoder='ImageNetCNN'
    ):
        if encoder == 'ImageNetCNN':
            self.encoder = create_imageNetCNN()
            self.output_dim = self.encoder.layers[-5].get_config()['filters']
        else:
            raise NotImplementedError()

        self._label_to_train_indices = {}
        self._proto_index_to_label = None
        self.prototypes = None

    def meta_train(
            self,
            meta_train_X,
            meta_train_Y,
            meta_val_X=None,
            meta_val_Y=None,
            n_episode=8000,
            n_way=60,
            ks_shots=5,
            kq_shots=5,
            optimizer='Adam',
            logging_interval=200
    ):
        """Trains the model on the meta-training set.

        Args:
            meta_train_X (numpy.array): The data set used for meta-training, must be of dimension 4.
            meta_train_Y (numpy.array): The corresponding set of labels, must be a single dimension array of integers
            meta_val_X (numpy.array): A validation set to track the performance of the model every few episodes.
            meta_val_Y (numpy.array): The corresponding set of validation labels.
            n_episode (int): Number of episodes for meta-training.
            n_way (int): Number of ways (or classes per episode).
            ks_shots (int): Number of image per class in the support set.
            kq_shots (int): Number of image per class in the query set.
            optimizer (tf.keras.optimizer): A valid Keras optimizer for training.
            logging_interval (int): log and display metrics every ‘logging_interval‘ episodes.

        Return:
            3-tuple:
                - acc (list(float)): history of accuracy logged every ‘logging_interval‘ episodes.
                - loss (list(float)): history of loss logged every ‘logging_interval‘ episodes.
                - test_acc (list(float)): history of vaidation accuracy logged every ‘logging_interval‘ episodes.
        """
        acc = []
        loss = []
        test_acc = []

        lr_schedule, optimizer = self._prepare_optimizer(n_episode, optimizer)

        for episode in range(n_episode):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                distrib = tf.transpose(
                    run_episode(meta_train_X, meta_train_Y, n_way, ks_shots, kq_shots, self.encoder)
                )

                # Compute the loss value for this episode.
                labels = np.array([[i] * kq_shots for i in range(n_way)]).flatten()

                loss_value = _compute_loss(distrib, labels, n_way)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.encoder.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

            # Log every ‘logging_interval‘ episodes.
            if episode % logging_interval == 0:
                curr_acc = np.sum(tf.argmax(distrib, axis=1).numpy() == labels) / len(labels)
                acc.append(float(curr_acc))
                loss.append(float(loss_value))

                print(
                    "Episode %d; lr: %.2e, training loss: %.4f, train accuracy: %.2f"
                    % (
                        episode,
                        lr_schedule if type(lr_schedule) is float else lr_schedule(episode),
                        float(loss_value),
                        float(curr_acc)
                    )
                )
        return acc, loss, test_acc

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

    def fit(self, train_X, train_Y):
        """Fits the model to the data.

        Computes the prototype of each class and the internal mapping between prototype index
        and label.

        Args:
            train_X (numpy.array): A 4-dimensional array.
            train_Y (numpy.array): The corresponding labels, a 1-dimensional array of integers.
        """
        n_labels = len(np.unique(train_Y))
        prototypes = np.zeros((n_labels, self.output_dim)).astype(np.float32)

        self._proto_index_to_label = np.zeros((n_labels,)).astype(np.int32)

        self._label_to_train_indices = {
            ind: np.argwhere(train_Y.flatten() == ind).flatten()
            for ind in np.unique(train_Y)
        }

        for i, label in enumerate(np.unique(train_Y)):
            prototypes[i, :] = tf.reduce_mean(
                self.encoder(train_X[self._label_to_train_indices[label], :, :, :]),
                axis=0
            )
            self._proto_index_to_label[i] = label

        self.prototypes = prototypes

    def predict(self, X):
        """Make predictions and return the inferred labels.

        Args:
            X (numpy.array): A 4-dimensional array.

        Returns:
            preds (numpy.array): The predicted label for each data point, as a 1-dimensional array of integers.
        """
        dists = euclidean_distance(self.prototypes, self.encoder(X).numpy())
        return self._proto_index_to_label[tf.argmin(dists, axis=0).numpy()]


def run_episode(episode_X, episode_y, n_way, ks_shots, kq_shots, encoder):
    """ Computes softmax of distances for one sampled episode.

    Given a set X of input images, their corresponding labels y, episode
    parameters, N-way, K-support shots, K-query shots and an encoder:
    Sample a support set and a query set according to the episode's parameters
    then computes the prototypes from the support set, and return for each image
    of the query set, the negative logsoftmax of the the (negative) distance to
    each prototype.

    Args:
        episode_X (numpy.array): The dataset to sample images from.
        episode_y (numpy.array): The corresponding labels.
        n_way (int): Number of ways (or classes per episode).
        ks_shots (int): Number of image per class in the support set.
        kq_shots (int): Number of image per class in the query set.
        encoder (tf.keras.Model): The encoder used to map the image onto the metric space.

    Returns:
        dist (numpy.array): the distance of each query datapoint to each prototype.
    """

    # Sample N-way, KS support shots, KS query shots
    classes = np.random.choice(np.unique(episode_y), n_way)

    support_indices = np.zeros((n_way, ks_shots)).astype(np.int32)
    query_indices = np.zeros((n_way, kq_shots)).astype(np.int32)
    for i in range(n_way):
        indices = np.random.choice(np.argwhere(episode_y == classes[i]).flatten(), ks_shots + kq_shots)
        support_indices[i, :] = indices[:ks_shots]
        query_indices[i, :] = indices[ks_shots:]

    # Forward support into encoder
    support_embeddings = tf.reshape(
        encoder(episode_X[tf.reshape(support_indices, (n_way * ks_shots,)), :, :, :]),
        (n_way, ks_shots, -1)
    )

    # Compute prototypes as mean of each class
    prototypes = tf.reduce_mean(support_embeddings, axis=1)

    # Forward pass on query set
    query_embeddings = tf.reshape(
        encoder(episode_X[tf.reshape(query_indices, (n_way * kq_shots,)), :, :, :]),
        (n_way * kq_shots, -1)
    )

    # Compute distances, log of opposite softmax
    distances = euclidean_distance(prototypes, query_embeddings)

    # neg_log_dist = distances + tf.math.log(tf.reduce_sum(tf.exp(-distances), 0))
    return tf.math.softmax(-distances, axis=0)


def _compute_loss(distrib, labels, n_way):
    y_true = tf.keras.utils.to_categorical(labels, num_classes=n_way)
    loss_value = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true, distrib))
    return loss_value
