import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import librosa


class NoisePerturbAugmentor(object):
    """用于添加背景噪声的增强模型

    :param min_snr_dB: 最小的信噪比，以分贝为单位
    :type min_snr_dB: int
    :param max_snr_dB: 最大的信噪比，以分贝为单位
    :type max_snr_dB: int
    :param noise_path: 噪声文件夹
    :type noise_path: str
    :param sr: 音频采样率，必须跟训练数据的一样
    :type sr: int
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_snr_dB=10, max_snr_dB=30, noise_path="dataset/noise", sr=16000, prob=0.5):
        self.prob = prob
        self.sr = sr
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self._noise_files = self.get_noise_file(noise_path=noise_path)

    # 获取全部噪声数据
    @staticmethod
    def get_noise_file(noise_path):
        noise_files = []
        if not os.path.exists(noise_path): return noise_files
        for file in os.listdir(noise_path):
            noise_files.append(os.path.join(noise_path, file))
        return noise_files

    @staticmethod
    def rms_db(wav):
        """返回以分贝为单位的音频均方根能量

        :return: 均方根能量(分贝)
        :rtype: float
        """
        mean_square = np.mean(wav ** 2)
        return 10 * np.log10(mean_square)

    def __call__(self, wav):
        """添加背景噪音音频

        :param wav: librosa 读取的数据
        :type wav: ndarray
        """
        if random.random() > self.prob: return wav
        # 如果没有噪声数据跳过
        if len(self._noise_files) == 0: return wav
        noise, r = librosa.load(random.choice(self._noise_files), sr=self.sr)
        # 噪声大小
        snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
        noise_gain_db = min(self.rms_db(wav) - self.rms_db(noise) - snr_dB, 300)
        noise *= 10. ** (noise_gain_db / 20.)
        # 合并噪声数据
        noise_new = np.zeros(wav.shape, dtype=np.float32)
        if noise.shape[0] >= wav.shape[0]:
            start = random.randint(0, noise.shape[0] - wav.shape[0])
            noise_new[:wav.shape[0]] = noise[start: start + wav.shape[0]]
        else:
            start = random.randint(0, wav.shape[0] - noise.shape[0])
            noise_new[start:start + noise.shape[0]] = noise[:]
        wav += noise_new
        return wav


class SpeedPerturbAugmentor(object):
    """添加随机语速增强

    :param min_speed_rate: 新采样速率下限不应小于0.9
    :type min_speed_rate: float
    :param max_speed_rate: 新采样速率的上界不应大于1.1
    :type max_speed_rate: float
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=3, prob=0.5):
        if min_speed_rate < 0.9:
            raise ValueError("Sampling speed below 0.9 can cause unnatural effects")
        if max_speed_rate > 1.1:
            raise ValueError("Sampling speed above 1.1 can cause unnatural effects")
        self.prob = prob
        self._min_speed_rate = min_speed_rate
        self._max_speed_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_speed_rate, self._max_speed_rate, self._num_rates, endpoint=True)

    def __call__(self, wav):
        """改变音频语速

        :param wav: librosa 读取的数据
        :type wav: ndarray
        """
        wav = wav.astype(np.int64)
        if random.random() > self.prob: return wav.astype(np.int16)
        if self._num_rates < 0:
            speed_rate = random.uniform(self._min_speed_rate, self._max_speed_rate)
        else:
            speed_rate = random.choice(self._rates)
        if speed_rate == 1.0: return wav.astype(np.int16)

        old_length = wav.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        wav = np.interp(new_indices, old_indices, wav)
        return wav.astype(np.int16)


class VolumePerturbAugmentor(object):
    """添加随机音量大小

    :param min_gain_dBFS: 最小增益
    :type min_gain_dBFS: int
    :param max_gain_dBFS: 最小增益大
    :type max_gain_dBFS: int
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_gain_dBFS=-15, max_gain_dBFS=15, prob=0.5):
        self.prob = prob
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS

    def __call__(self, wav):
        """改变音量大小

        :param wav: librosa 读取的数据
        :type wav: ndarray
        """
        wav = wav.astype(np.float64)
        if random.random() > self.prob:
            return wav.astype(np.int16)
        gain = random.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        wav *= 10.**(gain / 20.)
        return wav.astype(np.int16)

