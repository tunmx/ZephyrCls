from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from torch.utils.data import Dataset
from zephyrcls.data.transform import ImagePipeline
from zephyrcls.data.transform import AudioPipeline
from zephyrcls.data.tools.audio_load import read_wav_data
import torch

def _data_to_tensor(image: np.ndarray):
    height, width, _ = image.shape
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    # print(image.shape)
    return torch.tensor(image.astype(np.float32))

def _image_load(path:str) -> np.ndarray:
    x = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    return x

def _audio_load(path:str, sr=16000) -> np.ndarray:
    signal, framerate = read_wav_data(path)
    assert sr == framerate
    return signal


_data_load_mode = dict(
    image=_image_load,
    audio=_audio_load,
)


class ClassificationDatasetBase(Dataset, metaclass=ABCMeta):

    def __init__(self, data_folder, labels_path=None, mode='train', transform=None, is_show=False, m_type='image'):
        self.data_folder = data_folder
        self.is_show = is_show
        self.m_type = m_type
        if labels_path:
            self.labels_path = labels_path
        self.mode = mode
        if transform:
            self.transform = transform
        else:
            self.transform = ImagePipeline()
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
        load_method = _data_load_mode[self.m_type]
        x = load_method(image_path)
        x = self.transform(x, mode=mode)
        if self.m_type == "image":
            if not self.is_show:
                x = _data_to_tensor(x)
        else:
            x = torch.tensor(x)

        return x,  torch.tensor(label)
