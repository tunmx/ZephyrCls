import os
import click
import tqdm
from loguru import logger
from zephyrcls.data import get_dataset
from zephyrcls.model import build_model
from zephyrcls.data.transform import Pipeline
from zephyrcls.model.loss import get_loss_function
from zephyrcls.utils.cfg_tools import load_cfg
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch

__all__ = ['evaluate']


@click.command(help='Evaluate the trained model.')
@click.argument('config_path', type=click.Path(exists=True))
@click.option('-model_path', '--model_path', default=None, type=click.Path())
@click.option('-data', '--data', default=None, type=click.Path())
def evaluate(config_path, model_path, data):
    logger.info("evaluation")
    cfg = load_cfg(config_path)
    print(cfg)
    # build training model
    model_cfg = cfg.model
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, 'best_model.pth')
        assert os.path.exists(model_path), f'{model_path}: The model was not matched.'
    net = build_model(model_cfg.name, **model_cfg.option)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # build loss
    loss_func = get_loss_function(model_cfg.loss.name)
    logger.info(f"loss function : {model_cfg.loss.name}")

    # load data
    data_cfg = cfg.data
    if data:
        data_cfg.val.option.img_path = data
    # build val dataset
    transform = Pipeline(**data_cfg.pipeline)
    val_dataset = get_dataset(data_cfg.val.name, transform=transform, **data_cfg.val.option)
    val_batch_size = data_cfg.val.batch_size
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True,
                                num_workers=0)

    val_loss = 0.0
    val_acc = 0.0
    val_bar = tqdm.tqdm(val_dataloader)
    for step, data in enumerate(val_bar):
        val_images, val_labels = data
        outputs = net(val_images.to(device))
        loss = loss_func(outputs, val_labels.to(device))
        val_loss += loss.item()
        accuracy = Accuracy().to(device)(outputs.to(device), val_labels.to(device))
        val_acc += accuracy

    avg_loss = val_loss / len(val_bar)
    avg_acc = val_acc / len(val_bar)

    logger.info(f"avg loss: {avg_loss}")
    logger.info(f"acc loss: {avg_acc}")


if __name__ == '__main__':
    evaluate()
