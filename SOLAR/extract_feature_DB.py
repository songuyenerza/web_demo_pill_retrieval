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

# some conflicts between tensorflow and tensoboard 
# causing embeddings to not be saved properly in tb
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
def prepare_im(im, device_ = 'cpu'):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = color_norm(im, _MEAN, _SD)
        # im = trans(im)
        if device_ == 'gpu':
            im = torch.from_numpy(im).cuda()
        else:
            im = torch.from_numpy(im)
        im = im.unsqueeze(0)
        return im

def extract_feat(net, img_path, device_ = 'cpu'):
    img_ = cv2.imread(img_path)
    # print('img_', img_, img_.shape)
    img_query = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    im = img_query
    # im = np.stack((im,)*3, axis=-1)   
    center = im.shape
    im_ = cv2.resize(im, (320, int(center[0]*320/center[1])))
    # im = cv2.resize(im, (512,512))
    center = im_.shape

    w =  center[1] * 0.92
    h =  center[0] * 0.92
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    # im = im[int(y):int(y+h), int(x):int(x+w)]
    v_ = torch.zeros(net.meta['outputdim'])

    for i in range(0, 360, 20):
        im = ImageRotate(im_, angle= i)
        im = im[int(y):int(y+h), int(x):int(x+w)]

        im = im.astype(np.float32, copy=False)
        im = prepare_im(im, device_)

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
    v /= 18
    v = v.pow(1./1)
    v /= v.norm()
    return v

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract DB")
    parser.add_argument("-d", "--device", default= 'cpu')
    args = parser.parse_args()
    device_ = args.device
    net = load_network(network_name='resnet101-solar-best.pth', device_ = 'cuda')
    if device_ == "cuda":
        net.cuda()
    net.eval()
    print(net.meta_repr())
    

    folder_test = "/Users/sonnguyen/Desktop/AI/DATK/DATN_web/static/data_pill/test_logo/"
    folder_train = "/Users/sonnguyen/Desktop/AI/DATK/DATN_web/static/data_pill/train/"

    path_list = []
    X = []

    with open( folder_train + 'paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]
    print("creat_DB:==============", len(IMAGE_PATH_DB))
    for i in range(len(IMAGE_PATH_DB)):
        print(i, IMAGE_PATH_DB[i])
        start = time.time()
        path_list.append(IMAGE_PATH_DB[i])
        key_path = folder_train + IMAGE_PATH_DB[i]
        x= extract_feat(net, key_path, device_)
        x = x.numpy()
        # print(x.shape)

        npy_save = folder_train + IMAGE_PATH_DB[i][:-4] 
        # np.save(npy_save, x)
        X.append(x)

        print(time.time() - start)


    with open(folder_test + 'paths.txt','r') as read:
        reader = csv.reader(read)
        list_correct = []
        n = 0
        for row in reader:
            time0 = time.time()
            q = row[0]
            print(q)
            matchs = []
            dists_list = []
            # dict_data[q] = matchs
            query_path = folder_test + q
            feat_q = extract_feat(net, query_path, device_)
            feat_q = feat_q.numpy()

            for r in row[1:]:
                matchs.append(r)
            score_list = []
            for i in range(len(X)):
                sco = np.dot(X[i], feat_q.T)
                score = np.array(sco)
                score_list.append(score)

            # sim = np.dot(X, Q.T)
            # ranks = np.argsort(-sim, axis=0)

            dict_list_score = dict(zip(path_list, score_list))
            dict_list_score_sort = sorted(dict_list_score.items(), key=lambda x:-x[1])[:30]
            print("time:", time.time() - time0)
            # print(dict_list_score_sort)
            path_list_sort = []
            list_score_sort =[]
            for i in range(len(dict_list_score_sort)):
                path_list_sort.append(dict_list_score_sort[i][0])
                list_score_sort.append(dict_list_score_sort[i][1])

            _str = '|'.join([str(elem) for elem in path_list_sort])
            _sc = '|'.join([str(elem) for elem in list_score_sort])
            APP_CODE = 'SOLAR_global'
            rst = f"{n},{APP_CODE},{q}, {_str}, {_sc}"
            n += 1
            with open('SOLAR/PILL_searching_solar.csv', 'a') as f:
                f.write(f'{rst}\n')