import torch
import torch.nn as nn
from torch.nn import functional as F
import torchsummary
from modeling.resnet import resnet18, resnet34, resnet50
from torch.nn import Module
from config import Config as cfg


class CMA_CPFNet(Module):
    def __init__(self, in_dim):
        super(CMA_CPFNet, self).__init__()
        self.fusion = Fusion(in_dim)

    def forward(self, derm, clinic):
        feat = torch.cat([derm, clinic], dim=1)
        feat = self.fusion(feat)
        att = F.softmax(feat, dim=1)
        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        derm_refined = att_1 * derm
        clinic_refined = att_2 * clinic
        return derm_refined, clinic_refined


class CMA_CPFNetSplit(Module):
    def __init__(self, in_dim):
        super(CMA_CPFNetSplit, self).__init__()
        self.mlp21c_derm = Fusion21c(in_dim)
        self.mlp21c_clinic = Fusion21c(in_dim)

    def forward(self, derm, clinic):
        derm21c = self.mlp21c_derm(derm)
        clinic21c = self.mlp21c_clinic(clinic)
        feat = torch.cat([derm21c, clinic21c], dim=1)
        att = F.softmax(feat, dim=1)
        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        derm_refined = att_1 * derm
        clinic_refined = att_2 * clinic

        return derm_refined, clinic_refined


class Fusion_CPFCasede(Module):
    def __init__(self, in_dim):
        super(Fusion_CPFCasede, self).__init__()
        self.fusion1 = Fusion(in_dim)
        self.fusion2 = Fusion(in_dim)

    def forward(self, derm, clinic, hyper, is_first):
        feat1 = torch.cat([derm, clinic], dim=1)
        feat1 = self.fusion1(feat1)
        att1 = F.softmax(feat1, dim=1)
        att_11 = att1[:, 0, :, :].unsqueeze(1)
        att_21 = att1[:, 1, :, :].unsqueeze(1)
        fusion_current = att_11 * derm + att_21 * clinic
        if is_first:
            return fusion_current
        else:
            feat2 = torch.cat([fusion_current, hyper], dim=1)
            feat2 = self.fusion2(feat2)
            att2 = F.softmax(feat2, dim=1)
            att_12 = att2[:, 0, :, :].unsqueeze(1)
            att_22 = att2[:, 1, :, :].unsqueeze(1)
            fusion = att_12 * fusion_current + att_22 * hyper
            return fusion


class MultiTaskPredictHead(Module):
    def __init__(self, in_dim):
        super(MultiTaskPredictHead, self).__init__()
        self.number_of_classes = cfg.classnum
        self.predict = self.predict_layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.number_of_classes),
        )

    def forward(self, x):
        output = self.predict(x)
        return output


class Fusion(nn.Module):
    def __init__(self, a):
        super(Fusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2*a, out_channels=a, dilation=1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=a, out_channels=a//2, dilation=1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=a//2, out_channels=2, dilation=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.relu(feat)
        feat = self.conv2(feat)
        feat = self.relu(feat)
        feat = self.conv3(feat)
        return feat


class Fusion21c(nn.Module):
    def __init__(self, a):
        super(Fusion21c, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=a, out_channels=a//2, dilation=1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=a//2, out_channels=a//4, dilation=1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=a//4, out_channels=1, dilation=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.relu(feat)
        feat = self.conv2(feat)
        feat = self.relu(feat)
        feat = self.conv3(feat)
        return feat


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.network = "resnet18"
        self.fusion_method = 'cpfcsd'
        self.cma_method = 'cpfnet'

        if self.network == "resnet18":
            self.backbone_derm = resnet18(pretrained=True)
            self.backbone_clinic = resnet18(pretrained=True)
            self.backbone_hyper = resnet18(pretrained=True)
            self.in_c = [64, 64, 128, 256, 512]
        elif self.network == "resnet34":
            self.backbone_derm = resnet34(pretrained=True)
            self.backbone_clinic = resnet34(pretrained=True)
            self.backbone_hyper = resnet34(pretrained=True)
            self.in_c = [64, 64, 128, 256, 512]
        elif self.network == "resnet50":
            self.backbone_derm = resnet50(pretrained=True)
            self.backbone_clinic = resnet50(pretrained=True)
            self.backbone_hyper = resnet50(pretrained=True)
            self.in_c = [64, 256, 512, 1024, 2048]

        if self.cma_method == 'cpfnet':
            self.cma_module_1 = CMA_CPFNet(self.in_c[1])
            self.cma_module_2 = CMA_CPFNet(self.in_c[2])
            self.cma_module_3 = CMA_CPFNet(self.in_c[3])
            self.cma_module_4 = CMA_CPFNet(self.in_c[4])
        elif self.cma_method == 'cpfnetsplit':
            self.cma_module_1 = CMA_CPFNetSplit(self.in_c[1])
            self.cma_module_2 = CMA_CPFNetSplit(self.in_c[2])
            self.cma_module_3 = CMA_CPFNetSplit(self.in_c[3])
            self.cma_module_4 = CMA_CPFNetSplit(self.in_c[4])
        else:
            print("error!!!!!!!!!!!")

        if self.fusion_method == 'cpfcsd':
            self.fusion_module_1 = Fusion_CPFCasede(self.in_c[1])
            self.fusion_module_2 = Fusion_CPFCasede(self.in_c[2])
            self.fusion_module_3 = Fusion_CPFCasede(self.in_c[3])
            self.fusion_module_4 = Fusion_CPFCasede(self.in_c[4])
        else:
            print("error!!!!!!!!!!!")

        self.head_derm = MultiTaskPredictHead(self.in_c[4])
        self.head_clinic = MultiTaskPredictHead(self.in_c[4])
        self.head_hyper = MultiTaskPredictHead(self.in_c[4])
        self.head_fusion = MultiTaskPredictHead(3*self.in_c[4])

    def forward(self, clinic, derm):
        # x1: clinical x2: dermoscopic

        derm = self.backbone_derm.conv1(derm)
        derm = self.backbone_derm.bn1(derm)
        derm = self.backbone_derm.relu(derm)
        derm = self.backbone_derm.maxpool(derm)
        derm_1 = self.backbone_derm.layer1(derm)

        clinic = self.backbone_clinic.conv1(clinic)
        clinic = self.backbone_clinic.bn1(clinic)
        clinic = self.backbone_clinic.relu(clinic)
        clinic = self.backbone_clinic.maxpool(clinic)
        clinic_1 = self.backbone_clinic.layer1(clinic)

        derm_1, clinic_1 = self.cma_module_1(derm_1, clinic_1)
        hyper_1 = self.fusion_module_1(derm_1, clinic_1, None, is_first=True)

        derm_2 = self.backbone_derm.layer2(derm_1)
        clinic_2 = self.backbone_clinic.layer2(clinic_1)
        hyper_2 = self.backbone_hyper.layer2(hyper_1)

        derm_2, clinic_2 = self.cma_module_2(derm_2, clinic_2)
        hyper_2 = self.fusion_module_2(derm_2, clinic_2, hyper_2, is_first=False)

        derm_3 = self.backbone_derm.layer3(derm_2)
        clinic_3 = self.backbone_clinic.layer3(clinic_2)
        hyper_3 = self.backbone_hyper.layer3(hyper_2)

        derm_3, clinic_3 = self.cma_module_3(derm_3, clinic_3)
        hyper_3 = self.fusion_module_3(derm_3, clinic_3, hyper_3, is_first=False)

        derm_4 = self.backbone_derm.layer4(derm_3)
        clinic_4 = self.backbone_clinic.layer4(clinic_3)
        hyper_4 = self.backbone_hyper.layer4(hyper_3)

        derm_4, clinic_4 = self.cma_module_4(derm_4, clinic_4)
        hyper_4 = self.fusion_module_4(derm_4, clinic_4, hyper_4, is_first=False)

        derm = F.adaptive_avg_pool2d(derm_4, (1, 1))
        derm = torch.flatten(derm, 1)
        output_derm = self.head_derm(derm)

        clinic = F.adaptive_avg_pool2d(clinic_4, (1, 1))
        clinic = torch.flatten(clinic, 1)
        output_clinic = self.head_clinic(clinic)

        hyper = F.adaptive_avg_pool2d(hyper_4, (1, 1))
        hyper = torch.flatten(hyper, 1)
        output_hyper = self.head_hyper(hyper)

        fusion = torch.cat([derm, clinic, hyper], dim=1)
        output_fusion = self.head_fusion(fusion)

        return output_clinic, output_derm, output_hyper, output_fusion


if __name__ == '__main__':
    model = BaseNet()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model.cuda()
    torchsummary.summary(model, [(3, 224, 224), (3, 224, 224)])
