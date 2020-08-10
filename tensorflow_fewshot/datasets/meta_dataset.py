from abc import ABC, abstractmethod


class MetaDataset(ABC):

    @abstractmethod
    def get_one_episode(self, n_way: int, ks_shot: int, kq_shot: int) -> tuple:
        pass