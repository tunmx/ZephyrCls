import cv2
import numpy as np
import zephyrcls as zcls
import click
import os
from tqdm import tqdm
import time


def backend_matching(backend):
    table = dict(torch=zcls.ZEPHYR_BACKEND_TORCH, onnx=zcls.ZEPHYR_BACKEND_ONNXRUNTIME, mnn=zcls.ZEPHYR_BACKEND_MNN)

    return table[backend]

@click.command(help='Exec inference flow.')
@click.argument('backend', type=click.Choice(['torch', 'onnx', 'mnn']))
@click.option('-config', '--config', type=click.Path(), default=None, )
@click.option('-model_path', '--model_path', type=click.Path(), default=None, )
@click.option('-data', '--data', type=click.Path(), )
@click.option('-save_dir', '--save_dir', default=None, type=click.Path(), )
@click.option('-input_shape', '--input_shape', default=None, multiple=True, nargs=2, type=int)
@click.option("-show", "--show", is_flag=True, type=click.BOOL, )
def inference(backend, config, model_path, data, save_dir, input_shape, show):
    backend_tag = backend_matching(backend)
    if input_shape is not None:
        input_size = input_shape[0]
    else:
        input_size = (112, 112)
    data_list = list()
    if os.path.isdir(data):
        data_list = [os.path.join(data, item) for item in os.listdir(data) if
                     item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
    else:
        data_list.append(data)
    if backend_tag == zcls.ZEPHYR_BACKEND_TORCH:
        assert config is not None, 'use torch need input config.'
        cfg = zcls.load_cfg(config)
        model_cfg = cfg.model
        if model_path is None:
            model_path = os.path.join(cfg.save_dir, 'best_model.pth')
            assert os.path.exists(model_path), 'The model was not matched.'
        if input_shape is None:
            input_size = tuple(cfg.data.pipeline.image_size)
        infer = zcls.create_inference(zcls.ZEPHYR_BACKEND_TORCH)
        net = infer(weights_path=model_path, input_shape=input_size, model_name=model_cfg.name,
                    model_option=model_cfg.option)
        print(net)

    elif backend_tag == zcls.ZEPHYR_BACKEND_ONNXRUNTIME:
        return NotImplementedError('not implement backend.')
    else:
        return NotImplementedError('not implement backend.')

    assert net is not None, 'error'

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    sum_t = 0.0
    for path in tqdm(data_list):
        image = cv2.imread(path)
        image = cv2.resize(image, input_size)
        t = time.time()
        prod = net(image)
        ut = time.time() - t
        sum_t += ut
        print(prod)
        idx = np.argmax(prod)
        print(f"{idx} {prod[idx]}")
        if show:
            cv2.imshow('image', image)
            cv2.waitKey(0)
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, os.path.basename(path)), image)

    print(f"avg use time: {sum_t / len(data_list)}")


if __name__ == '__main__':
    inference()
