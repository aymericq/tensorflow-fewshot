import unittest
from numpy.random import normal
import numpy as np
from tensorflow_fewshot.models.utils import euclidean_distance


# Given
class TestUtils(unittest.TestCase):

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
