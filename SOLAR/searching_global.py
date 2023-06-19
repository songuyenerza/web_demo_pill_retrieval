import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import cv2
import torch
from torch.utils.model_zoo import load_url
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import torch.nn as nn
import csv

from SOLAR.solar_global.utils.networks import load_network

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass

_MEAN = [0.485, 0.456, 0.406]
_SD = [0.229, 0.224, 0.225]
_scale_list = [0.7071, 1.0, 1.4142]
# _scale_list = [2.0, 1.414, 1.0, 0.707, 0.5]
# _scale_list = [1.2, 1.3, 1.4142]


def ImageRotate(image, angle):
    h, w = image.shape[:2]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE)
    return image_rot

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
        im = torch.from_numpy(im)
        im = im.unsqueeze(0)
        return im

def extract_feat(net, img_path):
    # img_ = cv2.imread(img_path)
    img_ = img_path
    img_query = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    im = img_query
    # im = np.stack((im,)*3, axis=-1)   
    center = im.shape
    im = cv2.resize(im, (320, int(center[0]*320/center[1])))
    # im = cv2.resize(im, (512,512))
    center = im.shape
    w =  center[1] * 0.9
    h =  center[0] * 0.9
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    # im = im[int(y):int(y+h), int(x):int(x+w)]
    im = im.astype(np.float32, copy=False)
    im = prepare_im(im)

    v = torch.zeros(net.meta['outputdim'])
    
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

def extract_feat_18(net, img_path):
    # img_ = cv2.imread(img_path)
    img_ = img_path
    img_query = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    im = img_query
    # im = np.stack((im,)*3, axis=-1)   
    center = im.shape
    im_ = cv2.resize(im, (256, int(center[0]*256/center[1])))
    # im = cv2.resize(im, (512,512))
    center = im_.shape

    w =  center[1] * 0.92
    h =  center[0] * 0.92
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    im = im[int(y):int(y+h), int(x):int(x+w)]
    v_ = torch.zeros(net.meta['outputdim'])

    for i in range(0, 360, 40):
        im = ImageRotate(im_, angle= i)
        im = im[int(y):int(y+h), int(x):int(x+w)]

        im = im.astype(np.float32, copy=False)
        im = prepare_im(im)

        v = torch.zeros(net.meta['outputdim'])
        
        for s in _scale_list:
            if s == 1:
                _input_t = im.clone()
            else:
                _input_t = nn.functional.interpolate(im, scale_factor=s, mode='bilinear', align_corners=False)
            v += net(_input_t).pow(1).cpu().data.squeeze()
        v /= len(_scale_list)
        v_ += v
    v = v_
    v /= 9
    v = v.pow(1./1)
    v /= v.norm()
    return v




class search_solar():
    def __init__(self, network_name = 'resnet101-solar-best.pth', folder_train = "/Users/sonnguyen/Desktop/AI/DATK/data_pill/train/"):
        net = load_network(network_name=network_name, device_ = 'cpu')
        # net.cuda()
        net.eval()
        print(net.meta_repr())
        self.model = net

        path_list = []
        X = []

        with open( folder_train + 'paths.txt','r') as f:
            IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]
        print("creat_DB:==============", len(IMAGE_PATH_DB))
        for i in range(len(IMAGE_PATH_DB)):
            # print(i, IMAGE_PATH_DB[i])
            path_list.append(IMAGE_PATH_DB[i])
            key_path = folder_train + IMAGE_PATH_DB[i][:-4] + '.npy'
            x = np.load(key_path)
            X.append(x)
        self.X = X
        self.path_list = path_list

    def searching_img(self, query_path):
        net = self.model
        X = self.X
        path_list = self.path_list
        t0 = time.time()
        feat_q = extract_feat_18(net, query_path)
        print("time per img", time.time() - t0)
        feat_q = feat_q.numpy()
        path_list_final = []
        score_list = []
        for i in range(len(X)):
            sco = np.dot(X[i], feat_q.T)
            score = np.array(sco)
            if score > 0.5:
                score_list.append(score)
                path_list_final.append(path_list[i])

        dict_list_score = dict(zip(path_list_final, score_list))
        dict_list_score_sort = sorted(dict_list_score.items(), key=lambda x:-x[1])[:20]
        # print("time:", time.time() - time0)
        # print(dict_list_score_sort)
        path_list_sort = []
        list_score_sort =[]
        for i in range(len(dict_list_score_sort)):
            path_output = dict_list_score_sort[i][0] 
            if "rot0." in path_output or "rot1." in path_output or "rot2." in path_output:
                path_output = path_output[:-9] + '.png'
            if path_output not in path_list_sort:
                path_list_sort.append(path_output)
                list_score_sort.append(dict_list_score_sort[i][1])
            if len(path_list_sort) == 5:
                break
        # return dict(zip(path_list_sort, list_score_sort))
        return path_list_sort, list_score_sort

if __name__ == '__main__':

    net = search_solar()
    print(net.searching_img(query_path= "/Users/sonnguyen/Desktop/AI/DATK/data_pill/test_logo/グループ1/グループ1①_1_0.png"))
    exit()




   