import os
import click
import cv2
import tqdm
import shutil
from loguru import logger
from zephyrcls.data import get_dataset
from zephyrcls.model import build_model
from zephyrcls.data.transform import Pipeline
from zephyrcls.model.loss import get_loss_function
from zephyrcls.utils.cfg_tools import load_cfg
from torch.utils.data import DataLoader
import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

__all__ = ['evaluate']

classify = ("00_other", "01_black", "02_white", "03_red", "04_purple", "05_yellow", "06_gray", "07_blue", "08_green", "09_brown")

@click.command(help='Evaluate the trained model.')
@click.argument('config_path', type=click.Path(exists=True))
@click.option('-model_path', '--model_path', default=None, type=click.Path())
@click.option('-data', '--data', default=None, type=click.Path())
@click.option('-analyze_dir', '--analyze_dir', default=None, type=click.Path(), help="Directory for detailed analysis, including confusion matrix and misclassified samples.")
def evaluate(config_path, model_path, data, analyze_dir):
    logger.info("Evaluation starts")
    cfg = load_cfg(config_path)

    # Build training model
    model_cfg = cfg.model
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, 'best_model.pth')
        assert os.path.exists(model_path), f'{model_path}: The model was not matched.'
    net = build_model(model_cfg.name, **model_cfg.option)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # Build loss
    loss_func = get_loss_function(model_cfg.loss.name)
    logger.info(f"Loss function: {model_cfg.loss.name}")

    # Load data
    data_cfg = cfg.data
    if data:
        data_cfg.val.option.img_path = data

    # Build validation dataset
    transform = Pipeline(**data_cfg.pipeline)
    val_dataset = get_dataset(data_cfg.val.name, transform=transform, **data_cfg.val.option)
    val_batch_size = data_cfg.val.batch_size
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=0)

    # Metrics
    accuracy_metric = torchmetrics.Accuracy().to(device)
    precision_metric = torchmetrics.Precision().to(device)
    recall_metric = torchmetrics.Recall().to(device)

    val_loss = 0.0
    all_labels = []
    all_preds = []
    misclassified_samples = []

    for data in tqdm.tqdm(val_dataloader):
        val_images, val_labels = data
        outputs = net(val_images.to(device))
        loss = loss_func(outputs, val_labels.to(device))
        val_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(val_labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        misclassified = preds != val_labels.to(device)
        misclassified_samples.extend([(img, pred.item(), label.item()) for img, pred, label in zip(val_images[misclassified], preds[misclassified], val_labels[misclassified])])

        # Update metrics
        accuracy_metric.update(outputs, val_labels.to(device))
        precision_metric.update(outputs, val_labels.to(device))
        recall_metric.update(outputs, val_labels.to(device))

    # Calculate average metrics
    avg_loss = val_loss / len(val_dataloader)
    avg_acc = accuracy_metric.compute()
    avg_precision = precision_metric.compute()
    avg_recall = recall_metric.compute()

    logger.info(f"Avg Loss: {avg_loss}")
    logger.info(f"Accuracy: {avg_acc}")
    logger.info(f"Precision: {avg_precision}")
    logger.info(f"Recall: {avg_recall}")

    # Reset metrics
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()

    if analyze_dir:
        os.makedirs(analyze_dir, exist_ok=True)

        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(analyze_dir, 'confusion_matrix.png'))
        plt.close()

        # Save misclassified samples
        for idx, (img, pred, label) in enumerate(misclassified_samples):
            true_label_dir = os.path.join(analyze_dir, classify[label])
            pred_label = classify[pred]
            os.makedirs(true_label_dir, exist_ok=True)
            img_path = os.path.join(true_label_dir, f'pred_{pred_label}_{idx}_img.png')
            img_rgb = img.cpu().numpy().transpose((1, 2, 0))  # Convert to HWC
            swap = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            plt.imsave(img_path, swap)

if __name__ == '__main__':
    evaluate()
