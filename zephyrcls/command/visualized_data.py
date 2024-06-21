import os
import click
import tqdm
import cv2
from loguru import logger
from zephyrcls.data import get_dataset
from zephyrcls.utils.cfg_tools import load_cfg
from zephyrcls.data.transform import Pipeline
from torch.utils.data import DataLoader

__all__ = ['visualized']


@click.command(help='Show the datasets.')
@click.argument('config_path', type=click.Path(exists=True))
@click.option('-data', '--data', default=None, type=click.Path())
def visualized(config_path, data):
    logger.info("evaluation")
    cfg = load_cfg(config_path)
    print(cfg)

    # load data
    data_cfg = cfg.data
    if data:
        data_cfg.train.option.data_folder = data
    # build train dataset
    transform = Pipeline(**data_cfg.pipeline)
    dataset = get_dataset(data_cfg.train.name, transform=transform, **data_cfg.train.option)
    dataset.is_show = True
    batch_size = data_cfg.train.batch_size
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0)

    transform_data = tqdm.tqdm(dataloader)
    images_tensor, kps_tensor = next(iter(transform_data))
    images = images_tensor.numpy()
    for img in images:
        print(img.shape)
        cv2.imshow("img", img)
        cv2.waitKey(0)
