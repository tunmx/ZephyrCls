import os
import click

classify = ("0", "1", "2",)



@click.command(help='Generate data.')
@click.option('-data', '--data', type=click.Path(exists=True))
@click.option('-train_rate', '--train_rate', type=float, default=0.92)
def generate_txt(data, train_rate):
    dirs = [item for item in os.listdir(data) if os.path.isdir(os.path.join(data, item))]
    train_data = list()
    val_data = list()
    for idx, item in enumerate(dirs):
        assert item in classify
        images_dirs = os.path.join(data, item)
        label_idx = classify.index(item)
        print(images_dirs, label_idx)
        data_list = [os.path.join(item, img_name) for img_name in os.listdir(images_dirs) if
                     img_name.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
        total = len(data_list)
        train_num = int(total * train_rate)
        tmp_train_data = data_list[:train_num]
        tmp_val_data = data_list[train_num:]
        for t_data in tmp_train_data:
            train_data.append([t_data, label_idx])
        for t_data in tmp_val_data:
            val_data.append([t_data, label_idx])

        with open(os.path.join(data, 'train.txt'), 'w') as f:
            txt = ""
            for path, label_idx in train_data:
                txt += f"{path} {label_idx}\n"
            f.write(txt)


        with open(os.path.join(data, 'val.txt'), 'w') as f:
            txt = ""
            for path, label_idx in val_data:
                txt += f"{path} {label_idx}\n"
            f.write(txt)



if __name__ == '__main__':
    generate_txt()
