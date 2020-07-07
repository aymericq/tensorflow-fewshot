import tensorflow as tf
import numpy as np


class PrototypicalNetwork:
    """Implements Prototypical Networks from the original paper (Snell et al., 2017)."""

    def __init__(
            self,
            encoder='ImageNetCNN'
    ):
        if encoder == 'ImageNetCNN':
            self.encoder = _create_imageNetCNN()
            self.output_dim = self.encoder.layers[-5].get_config()['filters']
        else:
            raise NotImplementedError()

        self.label2proto_index = {}
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
        acc = []
        loss = []
        test_acc = []

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

        self.encoder.compile(optimizer)

        for episode in range(n_episode):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                distrib = tf.transpose(
                    _run_episode(meta_train_X, meta_train_Y, n_way, ks_shots, kq_shots, self.encoder)
                )

                # Compute the loss value for this episode.
                labels = np.array([[i] * kq_shots for i in range(n_way)]).flatten()
                y_true = tf.keras.utils.to_categorical(labels, num_classes=n_way)
                loss_value = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true, distrib))

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

    def train(self, train_X, train_Y):
        n_labels = len(np.unique(train_Y))
        prototypes = np.zeros((n_labels, self.output_dim)).astype(np.float32)

        self.label2proto_index = {
            ind: np.argwhere(train_Y.flatten() == ind).flatten()
            for ind in np.unique(train_Y)
        }

        for i, label in enumerate(np.unique(train_Y)):
            prototypes[i, :] = tf.reduce_mean(
                self.encoder(train_X[self.label2proto_index[label], :, :, :])
            )

        self.prototypes = prototypes

    def predict(self, X):
        dists = _euclidean_distance(self.prototypes, self.encoder(X).numpy())
        return tf.argmax(dists, axis=0).numpy()


def _create_imageNetCNN(nb_hidden_layers=4, nb_filters=64, output_dim=64):
    """Creates a Keras Sequential Model as described in Matching Nets paper (Vinyals et al., 2016)."""

    layers = []
    for i in range(nb_hidden_layers):
        layers.extend([
            tf.keras.layers.Conv2D(nb_filters, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        ])
    layers.append(tf.keras.layers.Flatten())
    return tf.keras.models.Sequential(layers)


def _run_episode(train_set_X, train_set_y, n_way, ks_shots, kq_shots, encoder):
    """
    Given a set X of input images, their corresponding labels y, episode
    parameters, N-way, K-support shots, K-query shots and an encoder:
    Sample a support set and a query set according to the episode's parameters
    then computes the prototypes from the support set, and return for each image
    of the query set, the negative logsoftmax of the the (negative) distance to
    each prototype.
    """
    im_width, im_height, im_n_channels = train_set_X.shape[-3:]

    # Sample N-way, KS support shots, KS query shots
    classes = np.random.choice(np.unique(train_set_y), n_way)
    support_indices = np.zeros((n_way, ks_shots)).astype(np.int32)
    query_indices = np.zeros((n_way, kq_shots)).astype(np.int32)

    for i in range(n_way):
        indices = np.random.choice(np.argwhere(train_set_y == classes[i]).flatten(), ks_shots + kq_shots)
        support_indices[i, :] = indices[:ks_shots]
        query_indices[i, :] = indices[ks_shots:]

    # Forward support into encoder
    support_embeddings = tf.reshape(
        encoder(train_set_X[tf.reshape(support_indices, (n_way * ks_shots,)), :, :, :]),
        (n_way, ks_shots, -1)
    )

    # Compute prototypes as mean of each class
    prototypes = tf.reduce_mean(support_embeddings, axis=1)

    # Forward pass on query set
    query_embeddings = tf.reshape(
        encoder(train_set_X[tf.reshape(query_indices, (n_way * kq_shots,)), :, :, :]),
        (n_way * kq_shots, -1)
    )

    # Compute distances, log of opposite softmax
    distances = _euclidean_distance(prototypes, query_embeddings)

    # neg_log_dist = distances + tf.math.log(tf.reduce_sum(tf.exp(-distances), 0))
    return tf.math.softmax(-distances, axis=0)


def _euclidean_distance(prototypes, embeddings):
    """Compute the distance of each embedding to each prototype."""

    expanded_prototypes = tf.expand_dims(prototypes, 1)
    return tf.sqrt(tf.reduce_sum(
        tf.square(expanded_prototypes - embeddings),
        2
    ))
