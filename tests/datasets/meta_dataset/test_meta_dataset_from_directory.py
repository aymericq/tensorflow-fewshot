import unittest
from tensorflow_fewshot.datasets import MetaDatasetFromDisk
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd, mkdir
from shutil import rmtree


class TestMetaDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.curr_dir = "/".join((getcwd(), 'test_dir'))
        mkdir(self.curr_dir)

        self.ref_meta_ds = [
            (4 * np.ones((2, 2, 3), dtype=np.uint8), 1),
            (6 * np.ones((2, 2, 3), dtype=np.uint8), 1),
            (5 * np.ones((2, 2, 3), dtype=np.uint8), 1),
            (54 * np.ones((2, 2, 3), dtype=np.uint8), 2),
            (52 * np.ones((2, 2, 3), dtype=np.uint8), 2),
            (50 * np.ones((2, 2, 3), dtype=np.uint8), 2),
            (54 * np.ones((2, 2, 3), dtype=np.uint8), 2),
            (128 * np.ones((2, 2, 3), dtype=np.uint8), 3),
            (255 * np.ones((2, 2, 3), dtype=np.uint8), 3),
            (245 * np.ones((2, 2, 3), dtype=np.uint8), 3),
        ]

        self.meta_dir = "/".join((self.curr_dir, "meta"))
        mkdir(self.meta_dir)
        lbl_count = {}
        for im, label in self.ref_meta_ds:
            if label not in lbl_count:
                lbl_count[label] = 0
                mkdir("/".join((self.meta_dir, str(label))))
            filepath = "/".join((self.meta_dir, str(label), str(lbl_count[label]+1) + ".jpg"))
            plt.imsave(filepath, im)
            lbl_count[label] += 1

        self.ref_train_ds = [
            (4  *np.ones((2, 2, 3), dtype=np.uint8), -11),
            (54 *np.ones((2, 2, 3), dtype=np.uint8), 2),
            (124*np.ones((2, 2, 3), dtype=np.uint8), 3),
            (255*np.ones((2, 2, 3), dtype=np.uint8), 3),
        ]

        self.train_dir = "/".join((self.curr_dir, "train"))
        mkdir(self.train_dir)
        lbl_count = {}
        for im, label in self.ref_train_ds:
            if label not in lbl_count:
                lbl_count[label] = 0
                mkdir("/".join((self.train_dir, str(label))))
            filepath = "/".join((self.train_dir, str(label), str(lbl_count[label]+1) + ".jpg"))
            plt.imsave(filepath, im)
            lbl_count[label] += 1

    def tearDown(self) -> None:
        rmtree(self.curr_dir)

    def test_create_and_get_meta_train_dataset_from_folder(self):
        # Given
        mds = MetaDatasetFromDisk(self.curr_dir)

        # When
        meta_train_generator = mds.get_meta_dataset_generator()
        meta_train_ds = list(meta_train_generator)

        labels = list(map(lambda x: x[1], meta_train_ds))
        imgs = list(map(lambda x: x[0], meta_train_ds))

        # Then
        self.assertEqual(len(self.ref_meta_ds), len(meta_train_ds))
        for im, label in zip(imgs, labels):
            im_in_ref_ds = False
            for ref_im, ref_label in self.ref_meta_ds:
                if (im == ref_im).all():
                    self.assertEqual(str(label), str(ref_label))
                    im_in_ref_ds = True

            self.assertTrue(im_in_ref_ds)

    def test_create_and_get_train_dataset_from_folder(self):
        # Given
        mds = MetaDatasetFromDisk(self.curr_dir)

        # When
        train_ds_generator = mds.get_train_dataset_generator()
        train_ds = list(train_ds_generator)

        labels = list(map(lambda x: x[1], train_ds))
        imgs = list(map(lambda x: x[0], train_ds))

        # Then
        self.assertEqual(len(self.ref_train_ds), len(train_ds))
        for im, label in zip(imgs, labels):
            im_in_ref_ds = False
            for ref_im, ref_label in self.ref_train_ds:
                if (im == ref_im).all():
                    self.assertEqual(str(label), str(ref_label))
                    im_in_ref_ds = True

            self.assertTrue(im_in_ref_ds)

    def test_one_episode_has_right_size(self):
        # Given
        mds = MetaDatasetFromDisk(self.curr_dir)
        n_way = 3
        kq_shot = 1
        ks_shot = 2

        # When
        support_generator, query_generator = mds.get_one_episode(n_way, ks_shot, kq_shot)
        support_set = list(support_generator)
        query_set = list(query_generator)

        # Then
        self.assertEqual(len(support_set), n_way*ks_shot)
        self.assertEqual(len(query_set), n_way * kq_shot)

    def test_one_episode_support_set_has_right_elements(self):
        # Given
        mds = MetaDatasetFromDisk(self.curr_dir)
        n_way = 3
        ks_shot = 2
        kq_shot = 1

        # When
        support_generator, query_generator = mds.get_one_episode(n_way, ks_shot, kq_shot)
        support_set = list(support_generator)

        # Then
        self.assertEqual(len(support_set), n_way*ks_shot)
        # Check if every elem in support set is in the original dataset with the right label
        for im, label in support_set:
            im_in_ref_ds = False
            for ref_im, ref_label in self.ref_meta_ds:
                if (im == ref_im).all():
                    self.assertEqual(label, str(ref_label))
                    im_in_ref_ds = True
            self.assertTrue(im_in_ref_ds)

    def test_one_episode_query_set_has_right_elements(self):
        # Given
        mds = MetaDatasetFromDisk(self.curr_dir)
        n_way = 3
        ks_shot = 0
        kq_shot = 2

        # When
        support_generator, query_generator = mds.get_one_episode(n_way, ks_shot, kq_shot)
        query_set = list(query_generator)

        # Then
        self.assertEqual(len(query_set), n_way*kq_shot)
        # Check if every elem in query set is in the original dataset with the right label
        for im, label in query_set:
            im_in_ref_ds = False
            for ref_im, ref_label in self.ref_meta_ds:
                if (im == ref_im).all():
                    self.assertEqual(label, str(ref_label))
                    im_in_ref_ds = True
            self.assertTrue(im_in_ref_ds)


if __name__ == '__main__':
    unittest.main()
