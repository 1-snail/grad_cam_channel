import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

import  matplotlib.pyplot as plt

class GradCamFeature:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)
        one_iamge_heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()

            # 最大最小归一化
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)    # 第 0 维增加维度
            for name, module in self.model.named_children():
                if name == "maxpool":
                    feature.register_hook(self.save_gradient)  # 保存梯度信息
                    self.feature = feature  # 保存该层输出特征图
                if name == 'fc':
                    feature = feature.view(feature.size(0), -1)  # 直接展平

                if name == 'classifier' :
                    feature = feature.view(feature.size(0), -1)   # 直接展平
                feature = module(feature)   # 正向传播
                if name == 'features':
                    feature.register_hook(self.save_gradient)    # 保存梯度信息
                    self.feature = feature    # 保存该层输出特征图
            classes = F.sigmoid(feature)     # feature 是分类输出的特征，  classes 是每个类别的概率
            one_hot, _ = classes.max(dim=-1)   # 取出最大概率
            self.model.zero_grad()   # 清空梯度
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            masks = F.relu((weight * self.feature)).squeeze(0)  # 将所有通道值相加，并去除第一维度
            for mask in masks:
                mask = cv2.resize(mask.data.cpu().numpy(), image_size)   # 向上采样到原图

                mask = mask - np.min(mask)
                if np.max(mask) != 0:
                    mask = mask / np.max(mask)

                heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
                cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))  # 原图和 热力图想叠加  img.transpose((1, 2, 0)) 将通道维度提到后面

                cam = cam - np.min(cam)
                if np.max(cam) != 0:
                    cam = cam / np.max(cam)

                one_iamge_heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
                #new_masks.append(heat_maps)
            one_iamge_heat_maps = torch.stack(one_iamge_heat_maps)
        return one_iamge_heat_maps

def draw_hotmap(featureImages, layerName):
    axis_number = int(np.sqrt(featureImages.shape[0]))
    fig, ax = plt.subplots(axis_number, axis_number)
    for i, feature_image in enumerate(featureImages):
        row = i // axis_number
        col = i - axis_number * (i // axis_number)
        feature_image = transforms.ToPILImage()(feature_image)

        ax[row][col].set_yticks([])
        ax[row][col].set_xticks([])

        ax[row][col].imshow(feature_image)


    fig.subplots_adjust(left=0.10, top=0.95, right=0.75, bottom=0.08, wspace=0.01, hspace=0.01)
    plt.savefig(str(layerName)+'FeatureMap.jpg', dpi=500, pad_inches=0, bbox_inches='tight')


