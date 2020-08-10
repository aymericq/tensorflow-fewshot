import numpy as np
from .meta_dataset import MetaDataset
from numpy.random import choice


class MetaDatasetFromArray(MetaDataset):

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.labels = np.unique(self.data_y)

    def get_one_episode(self, n_way: int, ks_shot: int, kq_shot: int) -> tuple:
        episode_labels = choice(self.labels, size=n_way, replace=False)
        support_indices = np.zeros((n_way * ks_shot,), dtype=np.int32)
        query_indices = np.zeros((n_way * kq_shot,), dtype=np.int32)
        for i_lbl, lbl in enumerate(episode_labels):
            lbl_indices = choice(
                np.argwhere(self.data_y == lbl).flatten(),
                size=(ks_shot + kq_shot),
                replace=False
            )
            support_indices[i_lbl * ks_shot:(i_lbl + 1) * ks_shot] = lbl_indices[:ks_shot]
            query_indices[i_lbl * kq_shot:(i_lbl + 1) * kq_shot] = lbl_indices[ks_shot:]

        support_set = self.data_x[support_indices, :, :, :], self.data_y[support_indices]
        query_set = self.data_x[query_indices, :, :, :], self.data_y[query_indices]

        return support_set, query_set
