#!/usr/bin/env python

from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from labelme import utils
import PIL.Image
import PIL.ImageDraw
import math
import numpy as np
import PIL.Image
import PIL.ImageDraw
from tools.rename_JPG_CR2 import get_filename_list
import os
import sys


def delete_chinese(check_str):
	out_str = ''
	for ch in check_str:
		if u'\u4e00' <= ch <= u'\u9fff' or ch == ' ':
			pass
		else:
			out_str += ch
	print(check_str)
	print(out_str)
	os.renames(check_str, out_str)



"""
import os
path = r"E:\1.PythonFile\CSU_BJ\data3\dataset\train\kjg"
count = 0
for root, dirs, files in os.walk(path):
    # print(files)
    for f in files:
        tmp_list = f.split(".")
        tmp_list[1] = "_kjg.jpg"
        tmp_list_1 = tmp_list[0].split("_")
        name = tmp_list_1[0] + "_" + tmp_list_1[0] + "_3"
        for i in range(1, len(tmp_list_1)):
            name = name + "_" + tmp_list_1[i]
        name += tmp_list[1]
        print(f, " ", name)
        count += 1
        os.rename(os.path.join(root, f), os.path.join(root, name))
print(count)

"""