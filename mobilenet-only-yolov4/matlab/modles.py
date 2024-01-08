#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolopred import YOLO

def matlabdet(img):
    yolo = YOLO()
    image = Image.open(img)
    r_image = yolo.detect_image(image)
    r_image.show()





if __name__ == "__main__":
    matlabdet('F:\studyp\mobilenet-only-yolov4\VOCdevkit\VOC2007\JPEGImages/0be36567-bee11e9c.jpg')