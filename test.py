import cv2
from zephyrcls.data import Pipeline

image = cv2.imread("/Users/tunm/Downloads/sn.jpeg")
pipeline = Pipeline()


for _ in range(100):
    image_trans = pipeline(image)

    cv2.imshow("w", image_trans)
    cv2.waitKey(0)