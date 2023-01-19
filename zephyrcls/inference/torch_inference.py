import cv2
import numpy as np
import torch
from .base import HamburgerABC
from zephyrcls.model import build_model
from loguru import logger
from zephyrcls.utils.image_tools import encode_images


class ClsInferenceTorch(HamburgerABC):

    def __init__(self, weights_path: str, model_name: str, model_option: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"input shape is {self.input_shape}")
        self.net = build_model(name=model_name, **model_option)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        logger.info("load weight successful.")

    @torch.no_grad()
    def _predict(self, data: torch.Tensor) -> torch.Tensor:
        result = self.net(data)

        return result

    def _postprocess(self, data: torch.Tensor):
        outputs = data.detach().numpy()
        # print(data)
        return outputs[0]


    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        assert len(
            bgr.shape) == 3, "The dimensions of the input image object do not match. The input supports a single image."
        image_resize = cv2.resize(bgr, self.input_shape)
        encode = encode_images(image_resize)
        encode = np.expand_dims(encode, 0)
        input_tensor = torch.Tensor(encode)

        return input_tensor

    def __str__(self):
        return str(self.net)
