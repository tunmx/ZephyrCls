from loguru import logger
import os
import tqdm
import json
from .base import ClassificationDatasetBase
import numpy as np


class TextLinesDataset(ClassificationDatasetBase):

    def _load_data(self, data_folder: str, labels_path: str) -> list:
        results = list()
        pairs_list = self._load_label(labels_path)
        logger.info(f"[{self.mode}]Data Loading...")
        for idx, (path, label) in enumerate(tqdm.tqdm(pairs_list)):
            full_path = os.path.join(self.data_folder, path)
            dic = dict(image=full_path, label=label)
            results.append(dic)

        return results

    def _load_label(self, path: str) -> list:
        pairs_list = list()
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                image_path, label_index = line.split(" ")
                pairs_list.append([image_path, int(label_index)])

        return pairs_list