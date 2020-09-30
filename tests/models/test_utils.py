import unittest
from numpy.random import normal
import numpy as np
from tensorflow_fewshot.models.utils import euclidean_distance, create_standardized_CNN
import tensorflow as tf


# Given
class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(37)
        tf.random.set_seed(37)

    def test_l2_norm_shape_is_10_5_when_passed_10_prototypes_10_embeddings(self):
        prototypes = normal(size=(5, 64))
        query_embeddings = normal(size=(10, 64))

        # When
        distances = euclidean_distance(prototypes, query_embeddings)

        # Then
        assert distances.shape == (5, 10)

    def test_euclidean_distance_is_right_on_two_examples(self):
        # Given
        prototypes = np.ones((2, 64))
        prototypes[0, :] *= 1
        prototypes[1, :] *= 4
        query_embeddings = np.ones((2, 64))
        query_embeddings[0, :] *= 3
        query_embeddings[1, :] *= 5

        true_dists = np.zeros((2, 2))
        true_dists[0, 0] = 8 * 2  # sqrt(64 * (1-3)^2)
        true_dists[1, 0] = 8  # sqrt(64 * (4-3)^2)
        true_dists[0, 1] = 8 * 4  # sqrt(64 * (1-5)^2)
        true_dists[1, 1] = 8  # sqrt(64 * (4-5)^2)

        # When
        distances = euclidean_distance(prototypes, query_embeddings)

        # Then
        assert distances.shape == (2, 2)
        assert np.all(distances == true_dists)


    def test_encoder_output_shape_is_10_64_when_passed_10_images(self):
        # Given
        encoder = create_standardized_CNN((28, 28, 1))
        images = normal(size=(10, 28, 28, 1))

        # When
        embeddings = encoder(images)

        # Then
        assert embeddings.shape == (10, 64)

    def test_output_has_correct_size_when_using_head(self):
        # Given
        encoder = create_standardized_CNN((28, 28, 1), use_dense_head=True, output_dim=5)
        images = normal(size=(10, 28, 28, 1))

        # When
        embeddings = encoder(images)

        # Then
        assert embeddings.shape == (10, 5)

    def test_raise_error_if_CNN_created_with_head_but_not_size_is_passed(self):
        with self.assertRaises(ValueError):
            encoder = create_standardized_CNN((28, 28, 1), use_dense_head=True)
