# -*- coding: utf-8 -*-
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# data_dir = './train'
data_dir = '/data/cenzhaojun/dataset/256_ObjectCategories/train'
features_dir = './extrac'
# shutil.copytree(data_dir, os.path.join(features_dir, data_dir[2:]))


def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()
    np.savetxt(saved_path, y, delimiter=',')


if __name__ == '__main__':
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    files_list = []
    x = os.walk(data_dir)
    for path,d,filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))

    print(files_list)
    resnet50_feature_extractor = models.resnet50(pretrained=True)
    print(resnet50_feature_extractor)
    resnet50_feature_extractor.fc = nn.Linear(2048, 1024)
    torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = torch.cuda.is_available()

    for x_path in files_list:
        print("x_path" + x_path)
        file_name = x_path.split('/')[-1]
        fx_path = os.path.join(features_dir, file_name + '.txt')
        print(fx_path)
        try:
            extractor(x_path, fx_path, resnet50_feature_extractor, use_gpu)
        except Exception:
            print('go !')