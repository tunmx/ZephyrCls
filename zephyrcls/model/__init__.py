from .lite_model import LiteModel
from .r34_model import ResNet34Classifier
import torch.nn as nn

__all__ = ['build_model']

_models_list_ = dict(LiteModel=LiteModel, R34Model=ResNet34Classifier)


def build_model(name: str, **option) -> nn.Module:
    assert name in _models_list_.keys()
    return _models_list_[name](**option)
