# -*- coding: utf-8 -*-
# @Time    : 2021/5/27 10:08
# @Author  : JZT
# @Email   : 915681919@qq.com
# @File    : config.py
# @Software: PyCharm


class Config:
	classnum = 5
	gpu_device = "0"  # "0,1"
	train_batch_size = 32  # 32
	test_batch_size = 20    # 要求是测试集数量的因子
	num_workers = 8
	
	train_number_epochs = 200
	img_size = (224, 224)
	initial_learning_rate = 0.005  # 0.0005
	momentum = 0.9
	weight_decay = 1e-4  # 1e-4
	
	# network = "resnet18"
	# network = "resnet34"
	# network = "resnet50"
	# network = "resnet101"
	network = "MobileNetV2"
	# network = "Xception"
	# network = "drn_d_105"
	# network = "Inception_ResNetv2"
	# network = "Inceptionv4"
	
	cma_method = 'mlpcross'
	fusion_method = 'cpfcsd'
	
	# 测试的时候需要更新模型的路径
	base_name = network + "_" + r"2021-05-27-15_08_39.pth"
	one_base_name = network + "_" + "2021-05-27-13_58_49.pth"
	
	base_path = r'./weights'
	one_base_path = r'./weights'

	img_data_path = r"E:\gldw_zt"
	pos_data_path = r"E:\gldw_zt_pos"
	
	# lable_name = {"bus": 0, "car": 1, "moto": 2, "pedestrian": 3, "truck": 4}
	lable_name = {"false": 0, "true": 1}
	