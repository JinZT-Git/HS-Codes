from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2 as cv
from labelme import utils
import PIL.Image
import PIL.ImageDraw
import math
import numpy as np
import PIL.Image
import PIL.ImageDraw
from tools.rename_JPG_CR2 import get_filename_list

np.set_printoptions(threshold=np.inf)


def point2contour(polygon_list, cvimg):
	# lableme的多边形的点，转为opencv的轮廓画轮廓，并在cvimg图上画出来
	# 由点转换为轮廓
	contour = []
	for point in polygon_list:
		point = np.array(point).astype(np.int32)
		point = np.expand_dims(point, axis=1)
		contour.append(point)
	cvimg = cv.drawContours(cvimg, contour, -1, (0, 0, 255), 1)
	cv.namedWindow("cvimg_point2contour", cv.WINDOW_NORMAL)
	cv.imshow("cvimg_point2contour", cvimg)
	cv.waitKey(0)
	return cvimg


def point2rectangle(rectangle_list, cvimg):
	# lableme的矩形框的点，转为opencv的点画矩形框，并在cvimg图上画出来
	
	for rectangle in rectangle_list:
		x = []
		y = []
		for i, j in rectangle:
			x.append(round(i))
			y.append(round(j))
		print(x, y)
		cv.rectangle(cvimg, (min(x), min(y)), (max(x), max(y)), (0, 255, 255), thickness=1)
	
	cv.namedWindow("cvimg_point2rectangle", cv.WINDOW_NORMAL)
	cv.imshow("cvimg_point2rectangle", cvimg)
	cv.waitKey(0)
	return cvimg


def plot_PIL(img):
	print(type(img), img.dtype)
	print(img.shape)
	plt.imshow(img)
	plt.show()


def plot_PIL2CV(img):
	img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
	print(type(img), img.dtype)
	print(img.shape)
	cv.namedWindow("opencv", cv.WINDOW_NORMAL)
	cv.imshow("opencv", img,)
	cv.waitKey()


def shape_to_mask(img_shape, points, shape_type=None, line_width=1, point_size=1):
	mask = np.zeros(img_shape[:2], dtype=np.uint8)
	mask = PIL.Image.fromarray(mask)
	draw = PIL.ImageDraw.Draw(mask)
	xy = [tuple(point) for point in points]
	if shape_type == "circle":
		assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
		(cx, cy), (px, py) = xy
		d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
		draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
	elif shape_type == "rectangle":
		assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
		draw.rectangle(xy, outline=1, fill=1)
	elif shape_type == "line":
		assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
		draw.line(xy=xy, fill=1, width=line_width)
	elif shape_type == "linestrip":
		draw.line(xy=xy, fill=1, width=line_width)
	elif shape_type == "point":
		assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
		cx, cy = xy[0]
		r = point_size
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
	else:
		assert len(xy) > 2, "Polygon must have points more than 2"
		# outline 是边界数字， fill是内部填充
		draw.polygon(xy=xy, outline=1, fill=1)
	mask = np.array(mask, dtype=bool)
	return mask


def analysis_json(path):
	print("Json file path:", path)
	img_path = path.replace('data4', 'data6\\img')
	mask_path = path.replace('data4', 'data6\\mask')
	
	path2 = path.replace('data4', 'Final\\final')
	
	json_file = path   # 方框
	json_file2 = path2  # 轮廓
	
	
	try:
		data = json.load(open(json_file))
	except:
		print("encoding='UTF-8'")
		data = json.load(open(json_file, encoding='UTF-8'))
		
	try:
		data2 = json.load(open(json_file2))
	except:
		print("encoding='UTF-8'")
		data2 = json.load(open(json_file2, encoding='UTF-8'))

	img = utils.img_b64_to_arr(data['imageData'])
	cvimg = cv.cvtColor(img, cv.COLOR_RGB2BGR)
	polygon_list = []  # 所有多边形的点列表的集合 存放JZT标记
	polygon_list2 = []  # 所有多边形的点列表的集合 存放HK的标记
	rectangle_list = []  # 所有矩形框的点列表的集合
	
	for shape in data['shapes']:
		if shape['shape_type'] == "rectangle":
			rectangle_list.append(shape['points'])
	
	for shape in data2['shapes']:
		if shape['shape_type'] == "polygon" and shape['label'] == "lesion":
			polygon_list.append(shape['points'])
		if shape['shape_type'] == "polygon" and shape['label'] == "lesion-HK":
			polygon_list2.append(shape['points'])
	
	if len(polygon_list2) > 0:
		polygon_list = polygon_list2
	
	# cls表示掩码（0 or 1）的图
	cls = np.zeros(img.shape[:2], dtype=np.uint8)
	for point in polygon_list:
		point = np.array(point).astype(np.int32)
		mask = shape_to_mask(img.shape, point, 'polygon')
		cls[mask] = 255
	
	# 显示mask的图
	# cv.namedWindow("2-mask", cv.WINDOW_NORMAL)
	# cv.imshow("2-mask", cls)
	# cv.waitKey(0)
	
	for num, rectangle in enumerate(rectangle_list):
		x = []
		y = []
		for i, j in rectangle:
			x.append(round(i))
			y.append(round(j))
		temp_mask = cls[min(y):max(y), min(x):max(x)]
		temp_img = cvimg[min(y):max(y), min(x):max(x)]
		
		temp_mask_path = mask_path.split('.')[0] + '_' + str(num + 1) + '_segmentation.png'
		temp_img_path = img_path.split('.')[0] + '_' + str(num + 1) + '.jpg'
		
		print(temp_mask_path, temp_img_path)
		cv.imwrite(temp_mask_path, temp_mask, [cv.IMWRITE_PNG_COMPRESSION, 10])
		cv.imwrite(temp_img_path, temp_img)
		

# print(mask.shape)
# # print(mask)
# print(np.max(mask))
# print(np.min(mask))

# class_id = np.asarray(class_id, np.uint8)  # [instance count,]
# lbl_names = list(lbl_names)
# print(type(lbl_names))
# print(len(lbl_names))
# class_name = lbl_names[1:]  # 不需要包含背景
#
# plt.subplot(221)
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(lbl_viz)
#
# plt.subplot(223)
# plt.imshow(mask, 'gray')
# plt.title(class_name[0]+'\n id '+str(class_id[0]))
# plt.axis('off')
# plt.show()


if __name__ == '__main__':
	root_path = r"E:\Dataset\XiangyaDerm\segmentation\data4"
	files_path_list_1, files_root_list_1, files_name_list_1, dir_list_1 = get_filename_list(root_path)
	
	
	# path = r"E:\Dataset\XiangyaDerm\segmentation\data4\BCC\BCC_60岁 3年.json"
	# analysis_json(path)
	# exit()
	
	for k, file in enumerate(files_path_list_1):
		if file[-4:] == 'json':
			print(k)
			# json_path = r"E:\2.PythonFile\mutl_task\tools\NEV_IMG.json"
			analysis_json(file)
			# break
	

# label_list = [x.replace('img', 'mask').split('.')[0] + '_segmentation.png' for x in img_list]
#
# elif self.mode == 'val' or self.mode == 'test':
# 	fold_s = fold[k_fold_test - 1]
# 	img_list = glob.glob(os.path.join(image_path, fold_s) + '/*.jpg')
# 	label_list = [x.replace('img', 'mask').split('.')[0] + '_segmentation.png' for x in img_list]
# return img_list, label_list


# path = "E:\Dataset\ISIC_2017\ISIC-2017_Test_v2_Part1_GroundTruth\ISIC_0012086_segmentation.png"
# path = "D:123.png"
# path2 = "D:12310.png"
#
# img1 = Image.open(path)
# img1 = np.array(img1)
#
# img2 = Image.open(path)
# img2 = np.array(img2)
# img1_2 = (img1 == img2)
#
# d = img1_2.any()  # 只要有一个TRUE，就返回TRUE
# e = img1_2.all()  # 必须都是True，才返回True，否则返回False
# print(d, e)
#
# img1[img1 == 0] = -1
# img1[img1 == 255] = -1
# print(img1)
# print(np.sum(img1))
#
# img1 = Image.open(path1)
# img2 = Image.open(path2)
#
# if img1 == img2:
# 	print("相同 1")
#
# img1 = np.array(img1)
# img2 = np.array(img2)
#
# if img1.any() == img2.any():
# 	print("相同 2")
# if img1.all() == 0 or img1.all() == 255:
# 	print(" 0 and 255")
#
# # print(img)
# print(img1.shape)
# print(np.max(img1))
# print(np.min(img1))
