import os
import cv2
import numpy as np
import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in folder:
        # print("check=====filename", filename)
        img = cv2.imread(filename)
        center = img.shape
        w =  center[1] * 0.98
        h =  center[0] * 0.98
        x = center[1]/2 - w/2
        y = center[0]/2 - h/2
        img = img[int(y):int(y+h), int(x):int(x+w)]
        images.append(img)
    return images

def load_images_name_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        images.append(filename)
    return images

def get_dimension(image):
    return image.shape

def get_mean(img):
    # mean_red = img[:,:,0].mean()
    # mean_blue = img[:,:,1].mean()
    # mean_green = img[:,:,2].mean()
    hist = cv2.calcHist([img], [0], None, [50],
        [0,180])
    hist = cv2.normalize(hist, hist).flatten()
    # return (mean_red, mean_blue, mean_green)
    return hist

def get_moment(img):
    moment_red = cv2.moments(img[:,:,0])['m00']
    moment_blue = cv2.moments(img[:,:,1])['m00']
    moment_green = cv2.moments(img[:,:,2])['m00']
    return (moment_red, moment_blue, moment_green)

def euclidean_distance(x=(0,0,0), y=(0,0,0)):
    return np.sqrt(np.square(x[0] - y[0]) 
                   + np.square(x[1] - y[1])
                   + np.square(x[2] - y[2]))
def query(query_img, DB): 
    # taking measures of the query image
    mean = get_mean(query_img)
    similarity_mean = DB['mean'].map(lambda x :cv2.compareHist(mean, x, cv2.HISTCMP_CORREL))
    similarity_mean = similarity_mean.to_dict()

    similar_images_mean = sorted(similarity_mean, key=similarity_mean.get, reverse = True)
    print(similarity_mean)

    # displaying the similar images
    # f, axarr = plt.subplots(rank, sharex=True, figsize=(5,30))
    # f.suptitle('Similar Images by Mean Color')
    similar_image = similar_images_mean[:5]
    # return imageDB.iloc[similar_image][4]
    return similar_image


dict_data = {}
with open('create_data_test_v2.csv','r') as read:
    reader = csv.reader(read)
    list_correct = []
    for row in reader:
        q = row[0]
        matchs = []
        for r in row[1:]:
            matchs.append(r)
        dict_data[q] = matchs
true = 0
folder_train = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/data_pill_1912/data_croped_2212/train/"
folder_save = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/data_pill_1912/data_croped_2212/check/"
with open('PILL_searching_solar_2012v3_sz320.csv','r') as read:
    reader = csv.reader(read)
    for row in reader:
        list_img_path = []
        query_path = row[2]
        # list_img_path.append(query_path)

        results = row[3]
        results = results.split('|')
        results = [name.strip(' ') for name in results]
        # results = [os.path.basename(name) for name in results]
        for key_path in results[:3]:
            list_img_path.append(folder_train + str(key_path))
        # print("check_list",list_img_path)

        images = load_images_from_folder(list_img_path)
        images_rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for image in images]
        imageDB = pd.DataFrame()
        imageDB['image_matrix'] = images_rgb
        imageDB['dimension'] = imageDB['image_matrix'].apply(get_dimension)
        imageDB['mean'] = imageDB['image_matrix'].apply(get_mean)
        imageDB['moment'] = imageDB['image_matrix'].apply(get_moment)
        imageDB['name '] = list_img_path
        result_Top = query(images_rgb[list_img_path.index(query_path)],imageDB)

        result_Top1 = result_Top[1]
        result_Top1 = imageDB.iloc[result_Top1][4]

        result_Top1 = os.path.basename(result_Top1)
        if "rot0." in result_Top1 or "rot1." in result_Top1 or "rot2." in result_Top1:
            result_Top1_ = result_Top1[:len(result_Top1) -13]
        else:
            result_Top1_ = result_Top1[:len(result_Top1) -8]

        query_name = os.path.basename(query_path)
        if result_Top1_ in dict_data[query_name[:-4]]:
            true +=1
        else:
            if result_Top1 != os.path.basename(list_img_path[1]):
                print(query_name, "====", result_Top1, "====", os.path.basename(list_img_path[1]))
print(true)


