import unittest
from tensorflow_fewshot.datasets import MetaDataset
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd, mkdir
from shutil import rmtree

class TestMetaDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.curr_dir = "/".join((getcwd(), 'test_dir'))

        self.ref_ds = [
            (4  *np.ones((2, 2, 3), dtype=np.uint8), 1),
            (54 *np.ones((2, 2, 3), dtype=np.uint8), 2),
            (128*np.ones((2, 2, 3), dtype=np.uint8), 3),
            (255*np.ones((2, 2, 3), dtype=np.uint8), 3),
        ]

        mkdir(self.curr_dir)
        lbl_count = {}
        for im, label in self.ref_ds:
            if label not in lbl_count:
                lbl_count[label] = 0
                mkdir("/".join((self.curr_dir, str(label))))
            filepath = "/".join((self.curr_dir, str(label), str(lbl_count[label]+1) + ".jpg"))
            plt.imsave(filepath, im)
            lbl_count[label] += 1


    def tearDown(self) -> None:
        rmtree(self.curr_dir)


    def test_create_meta_dataset_from_folder(self):
        # Given
        mds = MetaDataset(self.curr_dir)

        # When
        meta_train_generator = mds.get_meta_train_generator()
        meta_train_ds = list(meta_train_generator)

        labels = list(map(lambda x: x[1], meta_train_ds))
        imgs = list(map(lambda x: x[0], meta_train_ds))

        # Then
        self.assertEqual(len(self.ref_ds), len(meta_train_ds))
        for im, label in zip(imgs, labels):
            im_in_ref_ds = False
            for ref_im, ref_label in self.ref_ds:
                if (im == ref_im).all():
                    self.assertTrue(str(label) == str(ref_label))
                    im_in_ref_ds = True

            self.assertTrue(im_in_ref_ds)


if __name__ == '__main__':
    unittest.main()
