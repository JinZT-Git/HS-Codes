# -*- coding: utf-8 -*-
# @Time    : 2021/6/12 13:01
# @Author  : JZT
# @Email   : 915681919@qq.com
# @File    : mytest.py
# @Software: PyCharm

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2


# img = Image.open(r"F:\23.jpg")
# # img = Image.open(r"F:\23.jpg").convert('RGB')
# print(img.size)
# print(np.array(img).shape)
# # print(np.array(img)[2, 3])
# # print(np.array(img)[2, 3, 0])
# # print(np.array(img)[2, 3, 1])
# # print(np.array(img)[2, 3, 2])
#
# img = transforms.ToTensor()(img)
# print(img.size())
# print(img.max(), img.min())
# print(img[:, 2, 3])


img = cv2.imread(r"F:\23.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(type(img), img.shape)
img = transforms.ToTensor()(img)
print(img.size())
print(img.max(), img.min())
print(img[:, 2, 3])
