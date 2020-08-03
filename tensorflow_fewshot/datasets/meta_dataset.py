from os import listdir
from matplotlib.pyplot import imread

class MetaDataset:

    def __init__(self, path):
        self.path = path
        self.labels = {}

        for label in listdir(self.path):
            self.labels[label] = [elem for elem in listdir("/".join((self.path, label)))]

    def get_meta_train_generator(self):
        def generator_factory(mds):
            for label in mds.labels:
                for im_path in mds.labels[label]:
                    yield (
                        imread("/".join((mds.path, str(label), im_path))),
                        label
                    )

        return generator_factory(self)
