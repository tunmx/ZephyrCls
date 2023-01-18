from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from torch.utils.data import Dataset
from data.transform import Pipeline
import torch

def _data_to_tensor(image: np.ndarray, label):
    height, width, _ = image.shape
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    # print(image.shape)
    return torch.tensor(image.astype(np.float32)),  torch.tensor(label)


class ClassificationDatasetBase(Dataset, metaclass=ABCMeta):

    def __init__(self, data_folder, labels_path=None, mode='train', transform=None, is_show=False):
        self.data_folder = data_folder
        self.is_show = is_show
        if labels_path:
            self.labels_path = labels_path
        self.mode = mode
        if transform:
            self.transform = transform
        else:
            self.transform = Pipeline()
        self.data_list = self._load_data(self.data_folder, self.labels_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self._get_one_data(idx, mode=self.mode)

    @abstractmethod
    def _load_data(self, data_folder: str, labels_path: str) -> list:
        pass

    @abstractmethod
    def _load_label(self, path: str):
        pass

    def _get_one_data(self, idx: int, mode) -> tuple:
        data = self.data_list[idx]
        image_path = data['image']
        label = data['label']
        x = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if mode == 'train':
            x = self.transform(x, mode=mode)
        if not self.is_show:
            x = _data_to_tensor(x, label)

        return x, label
