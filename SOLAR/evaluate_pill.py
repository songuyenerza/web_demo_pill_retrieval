import json
import os
import csv

dict_data = {}
with open('expected_result.csv','r') as read:
	reader = csv.reader(read)
	list_correct = []
	for row in reader:
		q = row[0]
		matchs = []
		for r in row[1:]:
			matchs.append(r)
		# if q == "グループ2②_1_0":
		# 	print("here 11111111111" )
		dict_data[q] = matchs

correct = 0
count = 0
count_notfound = 0 
countting = {}
countting["1"] = 0
countting["3"] = 0
countting["5"] = 0
score_list = []
list_correct = []
if os.path.exists('OK.txt'):
	os.remove('OK.txt')
with open('PILL_searching_solar.csv','r') as read:
	reader = csv.reader(read)
	for row in reader:
		try:
			query_path = row[2]
		except:
			continue
		query_file = os.path.basename(query_path)
		results = row[3]
		if results == ' No matches found':
			continue
		
		results = results.split('|')
		results = [name.strip(' ') for name in results]
		results = [os.path.basename(name) for name in results]
		# print("results " , results)
		scores_ = row[4]
		scores_ = scores_.split('|')
		scores_ = [name.strip(' ') for name in scores_]
		for k , r in enumerate(results):
			if "rot0." in r or "rot1." in r or "rot2." in r:
				
				results[k] = r[:len(r) -13]
			else:
				results[k] = r[:len(r) -8]
		
		# results = [name[:len(name) -8] for name in results]
		
		# results = [name.split("/")[-2] for name in results]
		search = False
		count_Ok  = 0
		id_check = query_file[:len(query_file)-4]
		print("id_check " , results)
		try:
			list_match = dict_data[id_check]
		
			# if float(scores_[0]) < 0.90:
			# 	continue
			id_train = ""
			print("=============================\n")
			if len(list_match) > 0 :
				count +=1
				top_k = -1
				
				if results[0] in list_match:
					countting["1"] += 1
				if results[0] in list_match or results[1] in list_match or results[2] in list_match:
					countting["3"] += 1
				if results[0] in list_match or results[1] in list_match or results[2] in list_match or results[3] in list_match or results[4] in list_match:
					countting["5"] += 1

		except:
			err= 0

# print("correct " ,correct, count, "Top1=", correct / count )
print("total_img = ", count)
# print(score_list, min(score_list), max(score_list))
print("result: " ,countting, "_TOP1 = ", countting["1"] / count , "_Top5 = ", countting['5']/count )

