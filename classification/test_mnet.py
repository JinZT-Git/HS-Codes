# -*- coding: utf-8 -*-
# @Time    : 2021/5/27 15:10
# @Author  : JZT
# @Email   : 915681919@qq.com
# @File    : test_mnet.py
# @Software: PyCharm


import os
import sys
import torch
from torch.utils.data import DataLoader
from downloader import DownloaderData
import numpy as np
from network import BaseNet
from config import Config as cfg


def test(net, test_loader, criterion, e):
	net.eval()
	correct = 0
	total = 0
	mat = np.zeros([cfg.classnum, cfg.classnum])
	avg_loss = 0
	test_loss_record = []
	
	for b, (img0, img1, label) in enumerate(test_loader):
		if torch.cuda.is_available() and len(cfg.gpu_device) >= 1:
			img0 = img0.cuda()
			img1 = img1.cuda()
			label = label.cuda()
		output_x, output_y, output_h, output_f = net(img0, img1)
		
		loss_x = criterion(output_x, label)
		loss_y = criterion(output_y, label)
		loss_h = criterion(output_h, label)
		loss_f = criterion(output_f, label)
		loss_all = loss_x + loss_y + loss_h + loss_f
		test_loss_record.append(loss_all.item())
		
		# 三分支融合的预测
		_, predicted = output_f.max(1)
		for i in range(len(label)):
			mat[label[i]][predicted[i]] += 1
		total += label.size(0)
		
		correct += predicted.eq(label).sum().item()
		avg_loss = np.mean(test_loss_record)
		sys.stdout.write("\r Test  Process:%d: %d/%d | Loss: %.3f | Acc: %.3f | (%d/%d)" %
		                 (e + 1, b + 1, len(test_loader), avg_loss, 100. * correct / total, correct, total))
		sys.stdout.flush()
	
	print()
	return avg_loss, 100. * correct / total, mat


if __name__ == "__main__":
	dataset_test = DownloaderData(mode='test')
	dataloader_test = DataLoader(dataset_test, batch_size=cfg.test_batch_size, shuffle=False,
	                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
	print("测试集数量大小：", len(dataset_test))
	print("测试集一个epoch的batch数量：", len(dataloader_test), "\n")

	os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_device
	net = BaseNet()
	if torch.cuda.is_available() and len(cfg.gpu_device) >= 1:
		print("torch.cuda.is_available() :", torch.cuda.is_available(), " Use :", cfg.gpu_device)
		net = torch.nn.DataParallel(net).cuda()
	else:
		print("Not use cuda!!!")
	criterion = torch.nn.CrossEntropyLoss()
	
	if True:
		print("\nStart test !!!")
		checkpoint = torch.load(os.path.join(cfg.base_path, cfg.base_name))
		print("Now loading weight: ", cfg.base_name)
		net.load_state_dict(checkpoint['state_dict'])
		avg_loss, test_acc, mat = test(net, dataloader_test, criterion, -1)
		print("整数表示：\n", mat.astype(int))
		
		np.set_printoptions(precision=3)
		print(np.expand_dims(mat.sum(axis=1), 1))
		print("%百分比表示：\n", mat / np.expand_dims(mat.sum(axis=1), 1) * 100)
		print("End test!!!")
