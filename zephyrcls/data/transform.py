import numpy as np
from .tools.audio_load import get_mfcc_feature, feature_fix
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Pipeline(object):

    def __init__(self):
        pass

    def _transform_one(self, image: np.ndarray, mode='train') -> np.ndarray:
        pass

    def __call__(self, data: np.ndarray, mode='train', *args, **kwargs) -> np.ndarray:
        return self._transform_one(data, mode=mode)


class AudioPipeline(Pipeline):
    def __init__(self, signal_size: int = 16000, sr: int = 16000, speed_perturb: tuple = None,
                 volume_perturb: tuple = None, sometimes_rate=0.5):
        super().__init__()
        from .tools.audio_augmented import SpeedPerturbAugmentor, VolumePerturbAugmentor
        self.signal_size = signal_size
        self.sr = sr
        self.augment_pipeline = list()
        if speed_perturb:
            self.speed_augmentor = SpeedPerturbAugmentor(speed_perturb[0], speed_perturb[1], prob=sometimes_rate)
            self.augment_pipeline.append(self.speed_augmentor)
        if volume_perturb:
            self.volume_augmentor = VolumePerturbAugmentor(volume_perturb[0], volume_perturb[1],
                                                           prob=sometimes_rate)
            self.augment_pipeline.append(self.volume_augmentor)

    def _transform_one(self, data: np.ndarray, mode='train') -> np.ndarray:
        assert len(data.shape) < 3
        if mode == 'train':
            for aug in self.augment_pipeline:
                data = aug(data[0])
                data = np.asarray([data])

        data_input = get_mfcc_feature(data, samplerate=self.sr)
        data_input = np.array(data_input, dtype=np.float)
        data_input = np.reshape(data_input, (data_input.shape[0], data_input.shape[1], 1))
        data_input = feature_fix(data_input)
        data_input = data_input.transpose(2, 0, 1)
        return data_input


class ImagePipeline(Pipeline):

    def __init__(self, image_size=(96, 96), sometimes_rate=0.5, crop_percent=(0, 0.1), flip_lr=0.2,
                 gaussian_blur=(0, 1.0), multiply=(0.25, 1.55), contrast_normalization=(0.9, 1.2),
                 gamma_contrast=(0.9, 1.2), scale_x=(1, 1.), scale_y=(1, 1.), translate_percent_x=(-0.1, 0.1),
                 translate_percent_y=(-0.1, 0.1), rotate=(-10, 10), shear=(-5, 5), order=(0, 1), cval=0,
                 mode="constant"):
        from imgaug import augmenters as iaa
        # 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
        super().__init__()
        sometimes = lambda aug: iaa.Sometimes(sometimes_rate, aug)

        self.val_seq = iaa.Sequential([
            iaa.Resize(image_size, ),
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

            iaa.Resize(image_size, ),
        ])

        self.seq_map = dict(train=self.train_seq, val=self.val_seq)

    def _transform_one(self, data: np.ndarray, mode='train') -> np.ndarray:
        aug_det = self.seq_map[mode].to_deterministic()
        img_aug = aug_det.augment_image(data)

        return img_aug
