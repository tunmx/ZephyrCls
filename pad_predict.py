import cv2
import onnxruntime as ort
import numpy as np
import os

classify = ("00_black",
            "01_grey",
            "02_blue",
            "03_green",
            "04_white",
            "05_purple",
            "06_red",
            "07_brown",
            "08_yellow",
            "09_pink",
            "10_orange")


def padding(image, target_size=192):
    # 获取图像的高度和宽度
    h, w = image.shape[:2]
    # 确定新的尺寸，使得图像居中
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    # resize图像
    resized_image = cv2.resize(image, (new_w, new_h))
    # 创建一个黑色的背景
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    # 计算padding的偏移量
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    # 将resize后的图像放置在背景中
    padded_image[top:top + new_h, left:left + new_w] = resized_image
    return padded_image


class ONNXClassifier:
    def __init__(self, model_path, input_shape, input_name='data', output_name='output'):
        """
        初始化ONNX分类模型

        :param model_path: ONNX模型文件的路径
        :param input_shape: 模型输入的形状
        :param input_name: 模型输入节点的名称
        :param output_name: 模型输出节点的名称
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.input_name = input_name
        self.output_name = output_name

        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image):
        """
        预处理输入图像

        :param image: 输入图像，numpy数组
        :return: 预处理后的图像
        """
        pad = padding(image)
        pad = pad.astype(np.float32).transpose(2, 0, 1) / 255.0
        if pad.shape != self.input_shape:
            raise ValueError(f"Input image shape should be {self.input_shape}, but got {pad.shape}")
        pad = np.expand_dims(pad, axis=0)  # 添加批次维度
        return pad

    def postprocess(self, output):
        """
        后处理模型输出

        :param output: 模型输出
        :return: 分类结果
        """
        probabilities = np.squeeze(output)
        print(probabilities)
        class_idx = np.argmax(probabilities)
        confidence = probabilities[class_idx]
        return class_idx, confidence

    def predict(self, image):
        """
        预测图像的类别

        :param image: 输入图像，numpy数组
        :return: 类别索引和置信度
        """
        preprocessed_image = self.preprocess(image)
        ort_inputs = {self.input_name: preprocessed_image}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        return self.postprocess(ort_outs[0])

if __name__ == '__main__':
    cls = ONNXClassifier("workspace/color_cls_r34_x192_exp0/model.onnx", (3, 192, 192))
    image = cv2.imread("/Users/tunm/datasets/train_data_01_cut_3cls/0/f0d14d20-1ad0-482a-a177-d3a5f1a50699.jpg")

    out = cls.predict(image)
    print(out)
    print(classify[out[0]])