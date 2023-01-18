from zephyrcls.data import get_dataset
from zephyrcls.model import build_model
from zephyrcls.trainer.task import TrainTask
from torch.utils.data import DataLoader

train_dataset = get_dataset("TextLinesDataset",
                            data_folder="/Users/tunm/datasets/ClsPadAlign/batch1",
                            labels_path="/Users/tunm/datasets/ClsPadAlign/batch1/val.txt", mode="train")

val_dataset = get_dataset("TextLinesDataset",
                            data_folder="/Users/tunm/datasets/ClsPadAlign/batch1",
                            labels_path="/Users/tunm/datasets/ClsPadAlign/batch1/val.txt", mode="val")

net = build_model("LiteModel", class_num=3, width_mult=0.35, last_channel=1280, pool_pad=3)
train_batch_size = 32
val_batch_size = 16

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=0, pin_memory=True)


lr_schedule_option = dict(name="ReduceLROnPlateau", mode="min", factor=0.5, patience=5, verbose=True)
optimizer_option = dict(name="SGD", lr=0.01)
task = TrainTask(model=net,
                 save_dir="tmp",
                 loss_func="ce_loss",
                 lr_schedule_option=lr_schedule_option,
                 optimizer_option=optimizer_option,
                 wandb_cfg=None)

task.training(train_loader, val_loader, epoch_num=10, is_save=True)