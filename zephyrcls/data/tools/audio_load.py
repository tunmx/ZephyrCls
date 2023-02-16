import wave
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta

import random

def feature_fix(x, max_length=100, size=39):
    """
    截长补短操作, 经过处理后的feature如果长度未超过或者达不到max_length则采用取长补短, 补的部分全为0代替
    :param x: 音频前处理特征tensor
    :param max_length: 限制长值 默认100 通常表达为3秒内才音频信息长度
    :param size: 特征维度长度 mfcc默认为39 frequency为200
    :return: 返回修复后的tensor numpy ndarray
    """
    f_len = x.shape[0]
    feature = np.zeros((max_length, size, 1))
    if f_len > max_length:
        feature = x[:max_length]
    else:
        feature[:f_len] = x
    return feature

def read_wav_data(filename):
    """读取一个wav文件，返回声音信号的时域谱矩阵和播放时间"""
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取帧速率
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    # wave_data = wave_data
    return wave_data, framerate


def get_mfcc_feature(wavsignal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                     nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
                     N=2):
    """
    获取音频的mfcc特征, 仅支持16khz的采样率.
    :param wavsignal: 一维的音频信号
    :param samplerate: 默认16khz采样率, 目前仅支持这个采样率
    :param winlen: 分析窗口的长度，以秒为单位。默认为0.025秒(25毫秒)
    :param winstep: 连续窗口之间以秒为单位的步骤。默认为0.01s(10毫秒)
    :param nfft: 快速傅里叶变换尺寸. 默认512.
    :param lowfreq: mel滤波器最低边带. 默认0.
    :param preemph: 预加重滤波器预处理, 默认是0.97。
    :param ceplifter: 最终倒向系数应用提升器。0代表不举重。默认是22。
    :param appendEnergy: 倒谱系数被替换为总帧能量的对数
    :return: 返回mfcc维度
    """
    feat_mfcc = mfcc(wavsignal[0], samplerate=samplerate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt,
                     nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph, ceplifter=ceplifter,
                     appendEnergy=appendEnergy)
    feat_mfcc_d = delta(feat_mfcc, N)
    feat_mfcc_dd = delta(feat_mfcc_d, N)
    # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature


