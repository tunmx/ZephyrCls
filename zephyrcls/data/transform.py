import cv2
import numpy as np
from imgaug import augmenters as iaa


class Pipeline(object):

    def __init__(self,
                 image_size=(224, 224),
                 sometimes_rate=0.5,
                 crop_percent=(0, 0.1),
                 flip_lr=0.2,
                 gaussian_blur=(0, 1.0),
                 multiply=(0.25, 1.55),
                 contrast_normalization=(0.9, 1.2),
                 gamma_contrast=(0.9, 1.2),
                 scale_x=(1, 1.),
                 scale_y=(1, 1.),
                 translate_percent_x=(-0.1, 0.1),
                 translate_percent_y=(-0.1, 0.1),
                 rotate=(-10, 10),
                 shear=(-5, 5),
                 order=(0, 1),
                 cval=0,
                 mode="constant",
                 padding_mode=False):

        if padding_mode:
            resize_aug = iaa.Sequential([
                iaa.Scale({"height": image_size[0], "width": "keep-aspect-ratio"}),
                iaa.PadToSquare(pad_mode="constant", pad_cval=0),
                iaa.Resize(image_size)
            ])
        else:
            resize_aug = iaa.Resize(image_size)

        # 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
        sometimes = lambda aug: iaa.Sometimes(sometimes_rate, aug)

        self.val_seq = iaa.Sequential([
            # iaa.Resize(image_size, ),
            resize_aug,
        ])

        self.train_seq = iaa.Sequential([  # 建立一个名为seq的实例，定义增强方法，用于增强
            sometimes(iaa.Crop(percent=tuple(crop_percent))),
            iaa.Fliplr(flip_lr),  # 对百分之五十的图像进行做左右翻转
            iaa.GaussianBlur(tuple(gaussian_blur)),  # 在模型上使用0均值1方差进行高斯模糊
            # iaa.Rotate((-30, 30)),
            iaa.Multiply(tuple(multiply)),  # 改变亮度, 不影响bounding box
            iaa.ContrastNormalization(tuple(contrast_normalization)),  # 对比度
            iaa.GammaContrast(tuple(gamma_contrast), per_channel=True),  # 随机颜色变换
            # iaa.Sequential([
            #         iaa.Dropout(p=0.005),  # 随机删除像素点
            #     ]),

            # 对一部分图像做仿射变换
            sometimes(iaa.Affine(
                scale={"x": tuple(scale_x), "y": tuple(scale_y)},  # 图像缩放为80%到120%之间
                translate_percent={"x": tuple(translate_percent_x), "y": tuple(translate_percent_y)},  # 平移±20%之间
                rotate=tuple(rotate),  # 旋转±45度之间
                shear=tuple(shear),  # 剪切变换±16度，（矩形变平行四边形）
                order=list(order),  # 使用最邻近差值或者双线性差值
                cval=cval,  # 全白全黑填充
                mode=mode  # 定义填充图像外区域的方法
            )),

            # iaa.Resize(image_size, ),
            resize_aug,
        ])

        self.seq_map = dict(train=self.train_seq, val=self.val_seq)

    def _transform_one(self, image: np.ndarray, mode='train') -> np.ndarray:
        aug_det = self.seq_map[mode].to_deterministic()
        img_aug = aug_det.augment_image(image)
        # print(img_aug.shape)
        # cv2.imshow("s", img_aug)
        # cv2.waitKey(0)

        return img_aug

    def __call__(self, image: np.ndarray, mode='train', *args, **kwargs) -> np.ndarray:
        return self._transform_one(image, mode=mode)
