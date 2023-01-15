from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from torch.utils.data import Dataset


class ClassificationDatasetBase(Dataset, metaclass=ABCMeta):

    def __init__(self, images_path, labels_path, mode='train', transform=None, is_show=False):
        self.images_path = images_path