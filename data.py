import cv2
import numpy as np

from data.dataset.textline_dataset import TextLinesDataset
from torch.utils.data.dataloader import DataLoader

batch_size = 32
dataset = TextLinesDataset(data_folder="/Users/tunm/datasets/ClsPadAlign/batch1",
                           labels_path="/Users/tunm/datasets/ClsPadAlign/batch1/train.txt", is_show=True, mode="train")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

x, y = next(iter(loader))
for idx in range(batch_size):
    print(x[idx].shape)
    cv2.imshow("w", x[idx].numpy().astype(np.uint8))
    print(y[idx])
    cv2.waitKey(0)