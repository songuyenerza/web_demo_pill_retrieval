import argparse
import os
import numpy as np
import cv2
import torch
from models.model import SOLAR_LOCAL
import torchvision
import torch.nn as nn

_MEAN = [0.485, 0.456, 0.406]
_SD = [0.229, 0.224, 0.225]
_scale_list = [0.7071, 1.0, 1.4142]




def color_norm(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
def prepare_im( im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = color_norm(im, _MEAN, _SD)
        # im = trans(im)
        im = torch.from_numpy(im).cuda()
        im = im.unsqueeze(0)
        return im

def extract_feat(net, img_path):
    img_ = cv2.imread(img_path, 0)
    # img_query = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    im = img_
    im = np.stack((im,)*3, axis=-1)
    center = im.shape
    im = cv2.resize(im, (1024, int(center[0]*1024/center[1])))
    center = im.shape
    w =  center[1] * 0.9
    h =  center[0] * 0.9
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    # im = im[int(y):int(y+h), int(x):int(x+w)]
    im = im.astype(np.float32, copy=False)
    im = prepare_im(im)

    v = torch.zeros(128)
    
    for s in _scale_list:
        if s == 1:
            _input_t = im.clone()
        else:
            _input_t = nn.functional.interpolate(im, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(_input_t).pow(1).cpu().data.squeeze()
    v /= len(_scale_list)
    v = v.pow(1./1)
    v /= v.norm()
    return v

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    solar_local = SOLAR_LOCAL(soa= True, soa_layers= '345')
    
    model_weight_path = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/SOLAR/solar_local/weights/local-solar-345-liberty.pth"
    state_dict = torch.load(model_weight_path)
    solar_local.load_state_dict(state_dict)
    solar_local = solar_local.to(device)
    solar_local.eval()

    # print(solar_local)

    img_path = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/SOLAR/data/test/roxford5k/jpg/trinity_000015.jpg"

    feat_query = extract_feat(solar_local, img_path)
    print(feat_query)