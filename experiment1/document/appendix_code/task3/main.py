import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

nlm_image=cv2.imread(str("nlm.png"), cv2.IMREAD_GRAYSCALE)
if nlm_image is None:
    print("Failed to read the image.")
print(nlm_image.shape)
#读取NLM后的图像

low_resolution_image=cv2.resize(nlm_image,(128,128), interpolation=cv2.INTER_AREA)
#将图像缩小为128*128

#最邻近邻插值放大图像
nearest_image=cv2.resize(low_resolution_image,(512,512), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("/home/sanxi/DIP/experiment1/task3/output/nearest_image.png", nearest_image)

#双线性插值放大图像
bilinear_image=cv2.resize(low_resolution_image,(512,512), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("/home/sanxi/DIP/experiment1/task3/output/bilinear_image.png", bilinear_image)

#双三次插值放大图像
bicubic_image=cv2.resize(low_resolution_image,(512,512), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("/home/sanxi/DIP/experiment1/task3/output/bicubic_image.png", bicubic_image)
