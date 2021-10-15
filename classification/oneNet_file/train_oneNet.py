import os
import sys
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from config import Config as cfg
import time
from oneNet_file.one_net import BaseOneNet
from oneNet_file.one_downloader import DownloaderData


def test(net, test_loader, criterion, e):
    net.eval()
    correct = 0
    total = 0
    mat = np.zeros([cfg.classnum, cfg.classnum])
    avg_loss = 0
    test_loss_record = []

    for b, (img, label) in enumerate(test_loader):
        if torch.cuda.is_available() and len(cfg.gpu_device) >= 1:
            img = img.cuda()
            label = label.cuda()
        out_pre = net(img)

        loss = criterion(out_pre, label)
        test_loss_record.append(loss.item())

        # 三分支融合的预测
        _, predicted = out_pre.max(1)
        for i in range(len(label)):
            mat[label[i]][predicted[i]] += 1
        total += label.size(0)

        correct += predicted.eq(label).sum().item()
        avg_loss = np.mean(test_loss_record)
        sys.stdout.write("\r Test  Process:%d: %d/%d | Loss: %.3f | Acc: %.3f | (%d/%d)" %
                         (e+1, b+1, len(test_loader), avg_loss, 100. * correct / total, correct, total))
        sys.stdout.flush()

    print()
    return avg_loss, 100. * correct / total, mat


def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    # lr = cfg.warmup_learning_rate
    # if epoch > 10:
    #     lr = cfg.initial_learning_rate / (2 ** (epoch // 50))
    lr = cfg.initial_learning_rate / (2 ** (epoch // 50))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


def train(net, dataloader_train, dataloader_val, criterion, optimizer):
    best_acc = 0
    for e in range(cfg.train_number_epochs):
        net.train()
        lr = adjust_learning_rate(optimizer, e)
        train_loss_record = []
        total = 0
        correct = 0
        for b, (img, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and len(cfg.gpu_device) >= 1:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            out_pre = net(img)
            loss = criterion(out_pre, label)
            loss.backward()
            optimizer.step()
            train_loss_record.append(loss.item())
            e_avg_loss = np.mean(train_loss_record)

            _, predicted = out_pre.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            sys.stdout.write("\r Train Process:%d: %d/%d | AvgLoss: %.3f | Acc: %.3f | (%d/%d) | %.3f" %
                             (e+1, b+1, len(dataloader_train), e_avg_loss, 100. * correct / total, correct, total, best_acc))
            sys.stdout.flush()
        print()
        
        avg_loss, test_acc, mat = test(net, dataloader_val, criterion, e)
        if test_acc > best_acc:
            best_acc = test_acc
            state = {'epoch': e + 1, 'state_dict': net.state_dict(), 'best_dice': best_acc}
            torch.save(state, os.path.join(cfg.one_base_path, cfg.one_base_name))
        print()


if __name__ == "__main__":
    dataset_train = DownloaderData(mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.train_batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    print("训练集数量大小：", len(dataset_train))
    print("训练集一个epoch的batch数量：", len(dataloader_train), "\n")

    dataset_test = DownloaderData(mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.test_batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    print("测试集数量大小：", len(dataset_test))
    print("测试集一个epoch的batch数量：", len(dataloader_test), "\n")

    dataloader_val = dataloader_test
    print("验证集使用测试集替代！")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_device
    net = BaseOneNet()
    if torch.cuda.is_available() and len(cfg.gpu_device) >= 1:
        print("torch.cuda.is_available() :", torch.cuda.is_available(), " Use :", cfg.gpu_device)
        net = torch.nn.DataParallel(net).cuda()
    else:
        print("Not use cuda!!!")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.initial_learning_rate, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    
    if True:
        print("\nStart train !!!")
        cfg.one_base_name = cfg.network + "_" + time.strftime("%Y-%m-%d-%H_%M_%S.pth", time.localtime())
        print("Now time one_base_name: ", cfg.one_base_name, "\n")
        train(net, dataloader_train, dataloader_val, criterion, optimizer)
        print("End train !!!")

    if True:
        print("\nStart test !!!")
        checkpoint = torch.load(os.path.join(cfg.one_base_path, cfg.one_base_name))
        print("Now loading weight: ", cfg.one_base_name)
        net.load_state_dict(checkpoint['state_dict'])
        avg_loss, test_acc, mat = test(net, dataloader_test, criterion, -1)
        print("整数表示：\n", mat.astype(int))

        np.set_printoptions(precision=3)
        print(np.expand_dims(mat.sum(axis=1), 1))
        print("%百分比表示：\n", mat / np.expand_dims(mat.sum(axis=1), 1) * 100)
        print("End test!!!")
