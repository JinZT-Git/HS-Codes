import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
from PIL import Image
from config import Config as cfg
import cv2


class DownloaderData(Dataset):
	def __init__(self, mode='train'):
		if mode == "train":
			self.data_path = os.path.join(cfg.img_data_path, "train")
		elif mode == "test":
			self.data_path = os.path.join(cfg.img_data_path, "test")
		else:
			print("训练/测试模式错误！！！")
			exit()

		self.lable_dict = cfg.lable_name
		
		self.file_path_l = []
		self.file_path_r = []
		self.labels_l = []
		
		dirs = os.listdir(self.data_path)
		
		for dir2 in dirs:
			path2 = os.path.join(self.data_path, dir2)
			for filename in os.listdir(path2):
				file_path = os.path.join(path2, filename)
				self.file_path_l.append(file_path)

		random.shuffle(self.file_path_l)

		for file_path in self.file_path_l:
			self.file_path_r.append(file_path)
			self.labels_l.append(self.lable_dict[file_path.split("\\")[-2]])

	def __getitem__(self, index):
		img = Image.open(self.file_path_r[index]).convert('RGB')
		img = self.transform_tr_train(img)
		label = self.labels_l[index]

		return img, label

	def __len__(self):
		return len(self.file_path_r)

	def transform_tr_train(self, sample):
		tmpWH = cfg.img_size[0]
		new_image = Image.new('RGB', (tmpWH, tmpWH), (0, 0, 0))  # 生成灰色图像
		W, H = sample.size
		if W > H:
			scale = W / tmpWH
			nH = int(H / scale)
			nW = tmpWH
		else:
			scale = H / tmpWH
			nW = int(W / scale)
			nH = tmpWH
		sample = sample.resize((nW, nH), Image.ANTIALIAS)
		new_image.paste(sample, ((tmpWH - nW) // 2, (tmpWH - nH) // 2))

		composed_transforms = transforms.Compose([
			# transforms.Pad(),
			transforms.ToTensor()
			# transforms.Normalize()
		])
		return composed_transforms(new_image)


if __name__ == "__main__":
	batch_size = 10
	num_workers = 5
	data = DownloaderData(mode='train')
	
	dataloader_train = DataLoader(
		data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False)
	
	print("训练集数量 ：", len(data))
	print("batch_size ：", batch_size)
	print("训练集一个epoch的batch数量：", len(dataloader_train))

	for i, (img, labels) in enumerate(dataloader_train):
		print(img.shape)
		print(labels.shape)

		img1 = img[0]
		label = labels[0]

		print(img1.dtype)
		print(label.dtype)

		print(img1.shape)
		print(label.shape)

		print(label)
		print(img1.max(), img1.min())
		break