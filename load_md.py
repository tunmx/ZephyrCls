import os
import click
from loguru import logger
from zephyrcls.utils.cfg_tools import load_cfg
from zephyrcls.model import build_model
import torch
import onnxsim
import onnx

def export_onnx(net, model_save, input_shape: tuple):
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[0], input_shape[1])
    )
    torch.onnx.export(
        net,
        dummy_input,
        model_save,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )
    logger.info("finished exporting onnx.")
    logger.info("start simplifying onnx.")
    input_data = {"data": dummy_input.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(model_save, input_data=input_data)
    if flag:
        onnx.save(model_sim, model_save)
        logger.info("simplify onnx successfully")
    else:
        logger.error("simplify onnx failed")
    logger.info(f"export onnx model to {model_save}")



cfg = load_cfg("config/uniform_cls_litemodel_1.0_x256.yml")
model_cfg = cfg.model
net = build_model(model_cfg.name, **model_cfg.option)
net.load_state_dict(torch.load("workspace/uniform_cls_litemodel_1_0_x256_exp1/best_model.pth", map_location='cpu'))
net.eval()

# export_onnx(net, "exp1.onnx", (224, 224, ), )

tensor = torch.randn(1, 3, 224, 224)
print(net(tensor))