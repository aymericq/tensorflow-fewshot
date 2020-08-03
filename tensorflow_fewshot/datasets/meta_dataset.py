from os import listdir
from matplotlib.pyplot import imread

class MetaDataset:

    def __init__(self, path):
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
             path (string): The path of the root folder of the dataset.
        """
        self.path = path
        self.meta_path = '/'.join((self.path, 'meta'))
        self.train_path = '/'.join((self.path, 'train'))
        self.meta_labels = {}
        self.train_labels = {}

        for label in listdir(self.meta_path):
            self.meta_labels[label] = [elem for elem in listdir("/".join((self.meta_path, label)))]

        for label in listdir(self.train_path):
            self.train_labels[label] = [elem for elem in listdir("/".join((self.train_path, label)))]

    def get_meta_dataset_generator(self):
        """Returns a generator of the whole meta-dataset.

        Returns:
            generator: a generator of tuples (image, label).
        """
        def generator_factory(mds):
            for label in mds.meta_labels:
                for im_path in mds.meta_labels[label]:
                    yield (
                        imread("/".join((mds.path, 'meta', str(label), im_path))),
                        label
                    )

        return generator_factory(self)

    def get_train_dataset_generator(self):
        """Returns a generator of the whole training dataset.

        Returns:
            generator: a generator of tuples (image, label).
        """
        def generator_factory(mds):
            for label in mds.train_labels:
                for im_path in mds.train_labels[label]:
                    yield (
                        imread("/".join((mds.path, 'train', str(label), im_path))),
                        label
                    )

        return generator_factory(self)
