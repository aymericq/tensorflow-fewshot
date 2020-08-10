from typing import Generator, Tuple
from os import listdir
from matplotlib.pyplot import imread
from numpy.random import choice
import numpy as np


class MetaDataset:

    def __init__(self, path: str):
        """Creates a meta-dataset from a file path.

        Creates a meta-dataset given of tree of folders of the following form:
        dataset/
            meta/
                label_a/
                    img_001.jpg
                    img_002.jpg
                    ...
                label_b/
                    img_257.jpg
                    ...
            train/
                label_1/
                    train_1.jpg
                    ...
                label_2/
                    train_6.jpg
                    ...

        Args:
             path (str): The path of the root folder of the dataset.
        """
        self.path = path
        self.meta_path = '/'.join((self.path, 'meta'))
        self.train_path = '/'.join((self.path, 'train'))
        self.meta_label_to_file_indices = {}
        self.train_label_to_file_list = {}
        self.meta_ds_filenames = []

        for label in listdir(self.meta_path):
            files = [elem for elem in listdir("/".join((self.meta_path, label)))]
            self.meta_label_to_file_indices[label] = list(range(
                len(self.meta_ds_filenames),
                len(self.meta_ds_filenames) + len(files)
            ))
            self.meta_ds_filenames.extend(files)

        for label in listdir(self.train_path):
            self.train_label_to_file_list[label] = [elem for elem in listdir("/".join((self.train_path, label)))]

    def get_meta_dataset_generator(self) -> Generator[Tuple[np.array, int], None, None]:
        """Returns a generator of the whole meta-dataset.

        Returns:
            generator: a generator of tuples (image, label).
        """

        for label in self.meta_label_to_file_indices:
            for im_index in self.meta_label_to_file_indices[label]:
                yield (
                    imread("/".join((self.path, 'meta', str(label), self.meta_ds_filenames[im_index]))),
                    label
                )

    def get_train_dataset_generator(self) -> Generator[Tuple[np.array, int], None, None]:
        """Returns a generator of the whole training dataset.

        Returns:
            generator: a generator of tuples (image, label).
        """

        for label in self.train_label_to_file_list:
            for im_path in self.train_label_to_file_list[label]:
                yield (
                    imread("/".join((self.path, 'train', str(label), im_path))),
                    label
                )

    def get_one_episode(self, n_way: int, ks_shot: int, kq_shot: int) -> tuple:
        """Samples one episode from the meta training set.

        Args:
            n_way (int): Number of classes to sample.
            ks_shot (int): number of shots in the support set.
            kq_shot (int): number of shots in the query set.

        Returns:
            - support_set_generator: a generator of the support set, yielding (image, label) tuples.
            - query_set_generator: a generator of the query set, yielding (image, label) tuples.
        """

        # bidirectional integer mapping for meta labels
        labels_to_class_indices = {lbl: i for i, lbl in enumerate(self.meta_label_to_file_indices.keys())}
        class_indices_to_labels = {i: lbl for lbl, i in labels_to_class_indices.items()}

        classes = choice(list(class_indices_to_labels.keys()), n_way, replace=False)
        support_set_indices = np.zeros((n_way * ks_shot, 2), dtype=np.int)
        query_set_indices = np.zeros((n_way * kq_shot, 2), dtype=np.int)

        for i_class, cls in enumerate(classes):
            class_indices = choice(
                self.meta_label_to_file_indices[class_indices_to_labels[cls]],
                ks_shot + kq_shot,
                replace=False
            )
            support_set_indices[i_class * ks_shot:(i_class + 1) * ks_shot, 0] = class_indices[:ks_shot]
            support_set_indices[i_class * ks_shot:(i_class + 1) * ks_shot, 1] = cls
            query_set_indices[i_class * kq_shot:(i_class + 1) * kq_shot, 0] = class_indices[ks_shot:]
            query_set_indices[i_class * kq_shot:(i_class + 1) * kq_shot, 1] = cls

        support_labels = [
            class_indices_to_labels[support_set_indices[i, 1]]
            for i in range(n_way * ks_shot)
        ]

        support_set = [
            (
                imread(
                    "/".join((self.path, 'meta', str(support_labels[i]), self.meta_ds_filenames[support_set_indices[i, 0]]))
                ),
                support_labels[i]
            )
            for i in range(n_way * ks_shot)
        ]

        query_labels = [
            class_indices_to_labels[query_set_indices[i, 1]]
            for i in range(n_way * kq_shot)
        ]

        query_set = [
            (
                imread(
                    "/".join((self.path, 'meta', str(query_labels[i]), self.meta_ds_filenames[query_set_indices[i, 0]]))
                ),
                query_labels[i]
            )
            for i in range(n_way * kq_shot)
        ]

        return support_set, query_set
