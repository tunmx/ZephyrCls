from zephyrcls.data.transform import AudioPipeline
from zephyrcls.data.dataset.textline_dataset import TextLinesDataset
from torch.utils.data import DataLoader

pipeline = AudioPipeline(speed_perturb=(0.9, 1.1), volume_perturb=(-15, 15))
dataset = TextLinesDataset("data/", "data/val.txt", transform=pipeline, m_type="audio", mode='train')
loader = DataLoader(dataset, batch_size=16)

for img_batch, label_batch in loader:
    print(img_batch.shape)
    print(label_batch.shape)
