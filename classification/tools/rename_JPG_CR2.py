import os
from rawkit.raw import Raw
from rawkit.options import WhiteBalance
from PIL import Image
import shutil


def CR2_ppm_jpg(file_list):
	# 输入文件完整路径列表
	# 将CR2格式的文件转换为ppm，再将ppm转为jpg
	count = 0
	for pth in file_list:
		if pth.split('.')[1] == 'CR2':
			count += 1
			print(count, pth)
			with Raw(filename=pth) as raw:
				raw.options.white_balance = WhiteBalance(camera=False, auto=True)
				raw.save(filename=pth.replace('CR2', 'ppm'))
				
				img = Image.open(pth.replace('CR2', 'ppm'))
				img.save(pth.replace('CR2', '.jpg'))
	print(count)


def move_CR2_ppm(file_list):
	# 输入文件完整路径列表
	# 将CR2、ppm格式的文件从文件夹中移除到指定文件夹
	for pth in file_list:
		if pth.split('.')[1] == 'CR2' or pth.split('.')[1] == 'ppm':
			print(pth)
			shutil.move(pth, 'E:\Dataset\XiangyaDerm\segmentation\Temp\zCR2_PPM2')


def JPG_jpg(path_list):
	# windows 不区分JPG和jpg
	# 改名
	sum = 0
	for p in path_list:
		print(p)
		sum += 1
		if p.split('.')[-1] == 'JPG':
			print(sum)
			img = Image.open(p)
			print(p.replace('JPG', 'jpg'))
			img.save(p.replace('JPG', 'jpg'))
	# 删除
	sum = 0
	for p in path_list:
		sum += 1
		if p.split('.')[-1] == 'JPG':
			print(sum)
			os.remove(p)
			

def get_filename_list(root_path):
	path_list = []
	root_list = []
	name_list = []
	dir_list = []
	for root, dir, files in os.walk(root_path):
		# root所指的是当前正在遍历的这个文件夹的本身的地址
		# dirs是一个list ，内容是该文件夹中所有的目录的名字(不包括子目录)
		# files同样是list, 内容是该文件夹中所有的文件(不包括子目录)
		for f in files:
			path_list.append(os.path.join(root, f))
			root_list.append(root)
			name_list.append(f)
			dir_list.append(root.split('\\')[-1])
	return path_list, root_list, name_list, dir_list


if __name__ == '__main__':
	root_path1 = r"E:\Dataset\XiangyaDerm\segmentation\1"
	root_path2 = r"E:\Dataset\XiangyaDerm\segmentation\2"
	test_path = r"E:\Dataset\XiangyaDerm\segmentation\Temp\BCC"
	
	files_path_list_1, files_root_list_1, files_name_list_1, dir_list_1 = get_filename_list(root_path1)
	files_path_list_2, files_root_list_2, files_name_list_2, dir_list_2 = get_filename_list(root_path2)
	# files_path_list_3, files_name_list_3, dir_list_3 = get_filename_list(test_path)
	
	print(len(files_name_list_1))
	print(len(files_name_list_2))
	
	for i, p in enumerate(files_name_list_1):
		if p in files_name_list_2:
			# print(p)
			json = p[:-3]+'json'
			# print(json)
			p = os.path.join(files_root_list_1[i], p)
			json = os.path.join(files_root_list_1[i], json)
			
			# print(p)
			# print(json)
			#
			# exit()
			shutil.move(files_path_list_1[i], files_path_list_1[i].replace('\\1\\', '\\3\\'))
			
			shutil.move(json, json.replace('\\1\\', '\\3\\'))
			print(i)
			







