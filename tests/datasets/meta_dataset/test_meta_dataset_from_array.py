from unittest import TestCase

import numpy as np
from tensorflow_fewshot.datasets import MetaDatasetFromArray
from numpy.random import normal, randint


class TestMetaDatasetFromArray(TestCase):
    def test_dataset_from_array_returns_episodes_of_correct_length(self):
        # Given
        n_samples = 346
        width = 21
        height = 34
        n_channel = 3
        data_x = normal(size=(n_samples, width, height, n_channel))
        data_y = np.tile(np.arange(10), n_samples // 10 + 1)[:n_samples]
        np.random.shuffle(data_y)
        n_way, ks_shot, kq_shot = 5, 4, 3

        # When
        meta_ds = MetaDatasetFromArray(data_x, data_y)

        # Then
        support_set, query_set = meta_ds.get_one_episode(n_way, ks_shot, kq_shot)
        support_x, support_y = support_set
        query_x, query_y = query_set
        self.assertEqual(n_way * ks_shot, support_x.shape[0])
        self.assertEqual(n_way * ks_shot, support_y.shape[0])
        self.assertEqual(n_way * kq_shot, query_x.shape[0])
        self.assertEqual(n_way * kq_shot, query_y.shape[0])

    def test_episode_data_has_matching_labels(self):
        # Given
        n_samples = 100
        width = 21
        height = 34
        n_channel = 3
        data_x = normal(size=(n_samples, width, height, n_channel))
        data_y = np.tile(np.arange(6), n_samples // 6 + 1)[:n_samples]
        np.random.shuffle(data_y)
        n_way, ks_shot, kq_shot = 5, 4, 3

        # When
        meta_ds = MetaDatasetFromArray(data_x, data_y)
        support_set, query_set = meta_ds.get_one_episode(n_way, ks_shot, kq_shot)
        support_x, support_y = support_set
        query_x, query_y = query_set

        # Then
        labeling_dict = {}
        for i_support in range(support_x.shape[0]):
            exists_in_ref_data = False
            for i in range(data_x.shape[0]):
                if np.all(data_x[i] == support_x[i_support, :, :, :]):
                    exists_in_ref_data = True
                    if support_y[i_support] in labeling_dict:
                        self.assertEqual(labeling_dict[support_y[i_support]], data_y[i])
                    else:
                        labeling_dict[support_y[i_support]] = data_y[i]
            self.assertTrue(exists_in_ref_data)

        labeling_dict = {}
        for i_query in range(query_x.shape[0]):
            exists_in_ref_data = False
            for i in range(data_x.shape[0]):
                if np.all(data_x[i] == query_x[i_query]):
                    if query_y[i_query] in labeling_dict:
                        self.assertEqual(labeling_dict[query_y[i_query]], data_y[i])
                    else:
                        labeling_dict[query_y[i_query]] = data_y[i]
                    exists_in_ref_data = True
            self.assertTrue(exists_in_ref_data)

    def test_no_two_episodes_have_too_many_samples_in_common(self):
        # Given
        n_samples = 100
        width = 21
        height = 34
        n_channel = 3
        data_x = normal(size=(n_samples, width, height, n_channel))
        data_y = np.tile(np.arange(6), n_samples // 6 + 1)[:n_samples]
        np.random.shuffle(data_y)
        n_way, ks_shot, kq_shot = 5, 4, 3

        # When
        meta_ds = MetaDatasetFromArray(data_x, data_y)
        support_set1, query_set1 = meta_ds.get_one_episode(n_way, ks_shot, kq_shot)
        support_set2, query_set2 = meta_ds.get_one_episode(n_way, ks_shot, kq_shot)
        support_x1, support_y1 = support_set1
        support_x2, support_y2 = support_set2
        query_x1, query_y1 = query_set1
        query_x2, query_y2 = query_set2

        # Then
        count_number_of_samples_in_common = 0
        for image1 in support_x1:
            for image2 in support_x2:
                if np.all(image1 == image2):
                    count_number_of_samples_in_common += 1

        self.assertLess(count_number_of_samples_in_common, 0.5 * n_samples)

    def test_episode_has_same_number_of_sample_for_each_class(self):
        # Given
        n_samples = 100
        width = 21
        height = 34
        n_channel = 2
        data_x = normal(size=(n_samples, width, height, n_channel))
        data_y = np.tile(np.arange(6), n_samples // 6 + 1)[:n_samples]
        np.random.shuffle(data_y)
        n_way, ks_shot, kq_shot = 5, 4, 3

        # When
        meta_ds = MetaDatasetFromArray(data_x, data_y)
        support_set, query_set = meta_ds.get_one_episode(n_way, ks_shot, kq_shot)
        support_x, support_y = support_set
        query_x, query_y = query_set

        # Then
        support_labels = np.unique(support_y)
        self.assertEqual(len(support_labels), n_way)

        for label in support_labels:
            self.assertEqual(len(np.argwhere(support_y == label)), ks_shot)

        query_labels = np.unique(query_y)
        self.assertEqual(len(query_labels), n_way)

        for label in query_labels:
            self.assertEqual(len(np.argwhere(query_y == label)), kq_shot)

    def test_episode_labels_are_contiguous_integers(self):
        # Given
        n_samples = 100
        width = 3
        height = 3
        n_channel = 3
        data_x = normal(size=(n_samples, width, height, n_channel))
        data_y = np.tile(np.arange(6), n_samples // 6 + 1)[:n_samples]
        np.random.shuffle(data_y)
        n_way, ks_shot, kq_shot = 3, 4, 5

        # When
        meta_ds = MetaDatasetFromArray(data_x, data_y)
        support_set, query_set = meta_ds.get_one_episode(n_way, ks_shot, kq_shot)
        support_x, support_y = support_set
        query_x, query_y = query_set

        # Then
        self.assertEqual(n_way, len(np.unique(support_y)))
        self.assertEqual(0, np.min(support_y))
        self.assertEqual(n_way - 1, np.max(support_y))

        self.assertEqual(n_way, len(np.unique(query_y)))
        self.assertEqual(0, np.min(query_y))
        self.assertEqual(n_way - 1, np.max(query_y))
