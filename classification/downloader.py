import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
from PIL import Image
from config import Config as cfg


def img_2_pos(name):
	num = name.count("Image")
	if num == 1:
		name = name.replace("Image", "Posture")
	else:
		print("Error: 图片到动作的名称转换错误！！！")
		exit(0)
	return name


class DownloaderData(Dataset):
	def __init__(self, mode='train'):
		if mode == "train":
			self.img_data_path = os.path.join(cfg.img_data_path, "train")
			self.pos_data_path = os.path.join(cfg.pos_data_path, "train")
		elif mode == "test":
			self.img_data_path = os.path.join(cfg.img_data_path, "test")
			self.pos_data_path = os.path.join(cfg.pos_data_path, "test")
		else:
			print("训练/测试模式错误！！！")
			exit()

		self.lable_dict = cfg.lable_name
		
		self.img_filepath_l = []
		self.img_filepath_r = []
		self.pos_filepath_r = []
		self.labels_l = []
		
		dirs = os.listdir(self.img_data_path)
		
		for dir2 in dirs:
			path2 = os.path.join(self.img_data_path, dir2)
			for filename in os.listdir(path2):
				filepath = os.path.join(path2, filename)
				self.img_filepath_l.append(filepath)
				
		random.shuffle(self.img_filepath_l)

		for img_file in self.img_filepath_l:
			pos_file = img_2_pos(img_file)
			self.img_filepath_r.append(img_file)
			self.pos_filepath_r.append(pos_file)
			self.labels_l.append(self.lable_dict[img_file.split("\\")[-2]])
			# print(img_file)
			# print(pos_file)
			# print()

	def __getitem__(self, index):
		img = Image.open(self.img_filepath_r[index]).convert('RGB')
		pos = Image.open(self.pos_filepath_r[index]).convert('RGB')
		label = self.labels_l[index]
		return self.transform_tr_train(img), self.transform_tr_train(pos), label

	def __len__(self):
		return len(self.img_filepath_r)

	def transform_tr_train(self, sample):
		composed_transforms = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor()
			# transforms.Normalize()
		])
		return composed_transforms(sample)


if __name__ == "__main__":
	batch_size = 10
	num_workers = 5
	data = DownloaderData()
	dataloader_train = DataLoader(
		data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False
	)
	print("训练集数量 ：", len(data))
	print("batch_size ：", batch_size)
	print("训练集一个epoch的batch数量：", len(dataloader_train))
	for i, (kjg_img, hw_img, labels) in enumerate(dataloader_train):

		print(kjg_img.shape)
		print(hw_img.shape)
		print(labels.shape)

		img1 = kjg_img[0]
		img2 = hw_img[0]
		label = labels[0]

		print(img1.dtype)
		print(img2.dtype)
		print(label.dtype)

		print(img1.shape)
		print(img2.shape)
		print(label.shape)
		print(label)

		print(img1.max(), img1.min())
		break