from ast import Break
import json
import os
import csv
import shutil
import random
import cv2
data_folder_test = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/data_pill_1912/data_1111_crop_2step/test_logo/"
data_folder_train = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/data_pill_1912/data_1111_crop_2step/train/"
data_folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/data_pill_1912/data_1111_crop_2step/check_searching/OK/"
data_folder_output1 = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/data_pill_1912/data_1111_crop_2step/check_searching/NG/"
dict_data = {}
with open('expected_result.csv','r') as read:
    reader = csv.reader(read)
    list_correct = []
    for row in reader:
        q = row[0]
        matchs = []
        for r in row[1:]:
            matchs.append(r)
        # print("matchss",matchs)
        dict_data[q] = matchs

correct = 0
count = 0
count_notfound = 0 
countting = {}
countting["1"] = 0
countting["3"] = 0
countting["5"] = 0
list_correct = []
if os.path.exists('OK.txt'):
    os.remove('OK.txt')
with open('PILL_searching_solar_2012v3_sz320.csv','r') as read:
    reader = csv.reader(read)
    for row in reader:
        try:
            query_path = row[2]
        except:
            continue
        # print(query_path, "checkkkkkkkkkk")
        # query_file = os.path.basename(query_path)
        query_file = os.path.basename(query_path)
        results = row[3]
        if results == ' No matches found':
            continue
        
        results = results.split('|')
        results = [name.strip(' ') for name in results]
        # results = [os.path.basename(name) for name in results]
        scores_ = row[4]
        scores_ = scores_.split('|')
        scores_ = [name.strip(' ') for name in scores_]

        # results = [name[:len(name) -8] for name in results]
        
        # results = [name.split("/")[-2] for name in results]
        search = False
        count_Ok  = 0
        id_check = query_file[:len(query_file)-4]
        # print("id_check " , id_check)
        list_match = dict_data[id_check]
        results_copy = results.copy()
        for k , r in enumerate(results):
            if "rot0." in r or "rot1." in r or "rot2." in r:
                r = r.split('/')[-1]
                results[k] = r[:len(r) -13]
            else:
                r = r.split('/')[-1]
                results[k] = r[:len(r) -8]
        print("list_match", list_match, results)
    
        # if float(scores_[0])> 0.5:
        # 	continue
        id_train = ""
        print("=============================\n")
        if len(list_match) > 0 :
            count +=1
            top_k = -1
            

            for i in range(len(results[:5])):
                    if results[i] in list_match:
                        top_k = i
                        break

            print("checkkkkkkkkk",id_check)
            folder_id = os.path.join(data_folder_output, id_check) + '_TOP__' +  str(top_k)
            # if not os.path.exists(folder_id):
            # 	os.mkdir(folder_id)
            
            # shutil.copy(data_folder_test + id_check,  os.path.join(folder_id, "aori_" + id_check.split('/')[1]))
            print(data_folder_test + id_check)
            img_query = cv2.imread(data_folder_test + query_path)
            img_query = cv2.resize(img_query, (300,300))
            image_done = img_query
            print("top_k", top_k)
            
            for nk , rs in enumerate(results_copy[:5]):
                base_tr  = os.path.basename(results[nk])
                # shutil.copy(data_folder_train + rs, os.path.join(folder_id, "top"+ str(nk) + "_" + rs.split('/')[1]))
                img_key = cv2.imread(data_folder_train + rs)
                img_key = cv2.resize(img_key, (300,300))

                if rs not in list_match:
                    img_key = cv2.rectangle(img_key, (0,0), (300,300), (0,0,0), 20)

                    image_done = cv2.hconcat([image_done, img_key])
                    # cv2.imwrite("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/docimagepy/docimagepy/out.png", image_done)
                # print("rs, l",rs, l)
                else:
                    # print("rs, l",rs, list_match)
                    img_key = cv2.rectangle(img_key, (50,50), (250,250), (255,255,255), 30)
                    image_done = cv2.hconcat([image_done, img_key])
                    # top_k = nk
                    # break
            if top_k != 0:
                cv2.imwrite(os.path.join(data_folder_output1, "top"+ str(top_k) + "_" + id_check + '.png'), image_done)
            else:
                cv2.imwrite(os.path.join(data_folder_output, "top"+ str(top_k) + "_" + id_check + '.png'), image_done)



# print("correct " ,correct, count )
# print("countting " ,countting )
# print(len(list_correct))	