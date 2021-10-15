import torch
import torch.nn as nn
from torch.nn import functional as F
import torchsummary
from torch.nn import Module
from modeling.resnet import resnet18, resnet34, resnet50, resnet101
from modeling.mobilenet import MobileNetV2
from modeling.xception import AlignedXception
from modeling.drn import drn_d_105
from modeling.inception_resnet_v2 import My_In_Res_V2
from modeling.inceptionv4 import My_In_V4

from config import Config as cfg


# from torchvision.models import


class MLPHead(Module):
	def __init__(self, in_dim, number_of_classes):
		super(MLPHead, self).__init__()
		self.predict = self.predict_layer = nn.Sequential(
			nn.Linear(in_dim, in_dim//2),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(in_dim//2, number_of_classes)
		)

	def forward(self, x):
		output = self.predict(x)
		return output


def get_backbone_c():
	network = cfg.network
	if network == "resnet18":
		backbone = resnet18(pretrained=True)
		last_c = 512
	elif network == "resnet34":
		backbone = resnet34(pretrained=True)
		last_c = 512
	elif network == "resnet50":
		backbone = resnet50(pretrained=True)
		last_c = 2048
	elif network == "resnet101":
		backbone = resnet101(pretrained=True)
		last_c = 2048
	elif network == "MobileNetV2":
		backbone = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
		last_c = 320
	elif network == "Xception":
		backbone = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
		last_c = 2048
	elif network == "drn_d_105":
		backbone = drn_d_105(BatchNorm=nn.BatchNorm2d, pretrained=True)
		last_c = 2048
	elif network == "Inception_ResNetv2":
		backbone = My_In_Res_V2()
		last_c = 1536
	elif network == "Inceptionv4":
		backbone = My_In_V4()
		last_c = 1536
		
	return backbone, last_c


class BaseOneNet(nn.Module):
	def __init__(self):
		super(BaseOneNet, self).__init__()
		self.backbone, self.last_c = get_backbone_c()
		self.pre_mlp = MLPHead(self.last_c, cfg.classnum)

	def forward(self, x):

		
		x = self.backbone(x)

		t = F.adaptive_avg_pool2d(x, (1, 1))
		t = torch.flatten(t, 1)
		
		pre = self.pre_mlp(t)
		return pre


if __name__ == '__main__':
	net = BaseOneNet()
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	net.cuda()
	torchsummary.summary(net, (3, 224, 224))
