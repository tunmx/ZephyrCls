from .dataset import TextLinesDataset
from .dataset.base import ClassificationDatasetBase
from .transform import ImagePipeline

_dataset_map_ = dict(TextLinesDataset=TextLinesDataset, )


def get_dataset(name: str, **option) -> ClassificationDatasetBase:
    assert name in _dataset_map_.keys()
    return _dataset_map_[name](**option)
