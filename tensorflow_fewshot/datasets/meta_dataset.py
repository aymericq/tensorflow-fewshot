from os import listdir
from matplotlib.pyplot import imread

class MetaDataset:

    def __init__(self, path):
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
        def generator_factory(mds):
            for label in mds.meta_labels:
                for im_path in mds.meta_labels[label]:
                    yield (
                        imread("/".join((mds.path, 'meta', str(label), im_path))),
                        label
                    )

        return generator_factory(self)

    def get_train_dataset_generator(self):
        def generator_factory(mds):
            for label in mds.train_labels:
                for im_path in mds.train_labels[label]:
                    yield (
                        imread("/".join((mds.path, 'train', str(label), im_path))),
                        label
                    )

        return generator_factory(self)
