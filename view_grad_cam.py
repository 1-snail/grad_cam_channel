import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, resnet50,convnext_base
import torchvision
from grad_cam_resnet_feature import GradCamFeature ,draw_hotmap


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Grad-CAM')
    parser.add_argument('--image_name', default='both.png', type=str, help='the tested image name')
    parser.add_argument('--save_name', default='grad_cam.png', type=str, help='saved image name')

    opt = parser.parse_args()

    IMAGE_NAME = opt.image_name
    SAVE_NAME = opt.save_name
    test_image = (transforms.ToTensor()(Image.open(IMAGE_NAME))).unsqueeze(dim=0)
    model = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    if torch.cuda.is_available():
        test_image = test_image.cuda()
        model.cuda()
    grad_cam = GradCamFeature(model)
    feature_images = grad_cam(test_image)
    draw_hotmap(feature_images,"resnet50")

