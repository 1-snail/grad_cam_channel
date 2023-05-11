import argparse
from datetime import date

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, resnet18,convnext_base

from grad_cam_resnet_feature import GradCamFeature ,draw_hotmap

import cv2
import  numpy as np
import matplotlib.pyplot  as plt
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Grad-CAM')
    parser.add_argument('--image_name', default='both.png', type=str, help='the tested image name')
    parser.add_argument('--save_name', default='grad_cam.png', type=str, help='saved image name')

    opt = parser.parse_args()

    IMAGE_NAME = opt.image_name
    SAVE_NAME = opt.save_name
    test_image = (transforms.ToTensor()(Image.open(IMAGE_NAME))).unsqueeze(dim=0)
    model = convnext_base()
    #model = resnet18(pretrained=True)
    # model = vgg19(pretrained=True)
    if torch.cuda.is_available():
        test_image = test_image.cuda()
        model.cuda()
    grad_cam = GradCamFeature(model)
    feature_images = grad_cam(test_image)#.squeeze(dim=0)
   #feature_images = feature_images.permute((0,2,3,1))
    draw_hotmap(feature_images,2)

