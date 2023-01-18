from .lite_model import LiteModel
import torch.nn as nn

__all__ = ['build_model']

_models_list_ = dict(LiteModel=LiteModel, )


def build_model(name: str, **option) -> nn.Module:
    assert name in _models_list_.keys()
    return _models_list_[name](**option)
