# -*- coding: utf-8 -*-
# @Time    : 2021/6/11 12:38
# @Author  : JZT
# @Email   : 915681919@qq.com
# @File    : use_track.py
# @Software: PyCharm
"""
	调试 track.py
"""


from os.path import join, normpath
from track import perform_tracking
import os

root = r"F:\2\true"

file_path_list = os.listdir(root)

for file_path in file_path_list[:30]:
	file_path = os.path.join(root, file_path)
	
	print(file_path)
	# file_path = r"utils\3.jpg"
	model_name = "III"
	framework_name = "TFLite"
	visualize = True
	store = True
	
	perform_tracking(video=False, file_path=normpath(file_path), model_name=model_name, framework_name=framework_name, visualize=visualize, store=store)

