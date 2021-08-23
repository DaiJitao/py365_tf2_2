#coding:utf-8

#获取特征：#标题&内容的品牌，车系个数;标题，内容分词处理；标题，内容文章长度；获取各PV，click

import time
import json
import logging
import codecs
from ctypes import *

from get_segment_data_local import get_seg_data
from main_seriesNer import getSeries 
from get_extend_feature import FeatureInfo
from get_extend_feature import get_query_feature 
import traceback

import sys
import os
sys.path.append("/data/home/ds/lihui/feature_store_rtype/libsrc/")
#sys.path.append("/data/home/ds/lihui/feature_store_rtype/libsrc/util/")
sys.path.append("/data/home/ds/chenyongsheng/feature_store_rtype/tool/nlp_series/util/")
sys.path.append("/data/home/ds/chenyongsheng/feature_store_rtype/tool/nlp_series/")
sys.path.append("/data/home/ds/chenyongsheng/feature_store_rtype/tool/nlp_series/new_series/src/")
sys.path.append("/data/home/ds/chenyongsheng/feature_store_rtype/tool/nlp_series/configs/")
from label_series_local import get_res
from tokenizer_local_py3 import Segment
#from segment_process_tools import *

data_path = "/data/home/ds/lihui/feature_store_rtype/conf/"
stopwords_filename = data_path+"stopwords.dat"
userdict_filename = data_path+"userdict.dat"

#segment = Segment(stopwords_filename, userdict_filename)
#segment.loadstopwords()
lib=cdll.LoadLibrary("./jiebacpp/libsegment_python.so")
print('init: ' ,lib.jieba_qp_init(b"./jiebacpp/token_data/"))
jieba_qp_segment=lib.jieba_qp_segment
jieba_qp_segment.restype=c_char_p

max_len_title_seg,min_len_title_seg = 0,0
max_long_title_seg,min_long_title_seg = 0,0
max_len_content_seg,min_len_content_seg = 0,0
max_len_title,min_len_title = 0,0
max_len_long_title,min_len_long_title = 0,0
max_len_content,min_len_content = 0,0
max_len_series,min_len_series = 0,0
max_long_series,min_long_series = 0,0
max_content_series,min_content_series = 0,0
max_len_brand,min_len_brand = 0,0
max_long_brand,min_long_brand = 0,0
max_content_brand,min_content_brand = 0,0

def make_feature(data,other,biz_type,tag=1):
	"""tag=1:对标题提取特征
	tag=0:对内容提取特征，（车系单独处理）
	"""

	tmp_feature_dict = {
		"len_seg":0,
		"len_title":0,
		"len_series":0,
		"len_brand":0,
		'data_is_city':0,
		'data_is_series':0,
		'data_contain_series':0,
		'data_count_car_factory':0,
		'data_is_brand':0,
		'data_contain_brand':0,
		'data_is_factory':0,
		'data_contain_city':0,
		'data_is_price':0,
		'data_contain_price':0,
		'data_contain_factory':0
		}
	#分词特征
	if tag==0:   #不对content进行分词
		tmp_feature_dict["len_seg"] = '0'
	else:
		#tmp_feature_dict["len_seg"] = str(len(get_seg_data(segment, data).split("|")))
		#data=bytes(data, encoding='utf-8')
		#data=b(data)
		length=len(str(jieba_qp_segment(data.encode(encoding='utf-8'))).split('|'))-1
		if length < 1:
			length=0
		tmp_feature_dict["len_seg"] = str(length)

    #内容长度特征
	tmp_feature_dict["len_title"] = str(len(data))
	
	#车系品牌特征
	if tag == 0:
		#brand, series = getSeries(other,data,biz_type)
		series, brand = get_res(other,data,biz_type,len(data))
	else:
		#brand, series = getSeries(data,other,biz_type)
		series, brand = get_res(data,other,biz_type, len(data))
		t2=time.time()
		#print("series:", t2-t1)
		#return tmp_feature_dict
	brand_list = [item["id"] for item in brand]
	series_list = [item["id"] for item in series]
	tmp_feature_dict["len_brand"] = str(len(list(set(brand_list))))
	tmp_feature_dict["len_series"] = str(len(list(set(series_list))))

	#print("data is ->",data)
	data_is_series,\
	data_contain_series,\
	data_is_brand,\
	data_contain_brand,\
	data_is_factory,\
	data_contain_factory,\
	data_count_car_factory,\
	data_is_price,\
	data_contain_price,\
	data_is_city,\
	data_contain_city = get_query_feature(data)


	tmp_feature_dict["data_is_series"] = data_is_series
	tmp_feature_dict["data_contain_series"] = data_contain_series
	tmp_feature_dict["data_is_brand"] = data_is_brand
	tmp_feature_dict["data_contain_brand"] = data_contain_brand
	tmp_feature_dict["data_is_factory"] = data_is_factory
	tmp_feature_dict["data_contain_factory"] = data_contain_factory
	tmp_feature_dict["data_count_car_factory"] = data_count_car_factory
	tmp_feature_dict["data_is_price"] = data_is_price
	tmp_feature_dict["data_contain_price"] = data_contain_price
	tmp_feature_dict["data_is_city"] = data_is_city
	tmp_feature_dict["data_contain_city"] = data_contain_city
	
	
	return tmp_feature_dict

#min-max 标准化
def normalization(x):
	    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def get_data(filename, allsum):
	total_feature_dict = {}
	idx = 0
	f_in = open(filename,'r')
	for line in f_in:
		#query_id,query,title,long_title,content,publish_time,author,base_weight,quality_score,qanchor_list,key_words_list,interval_day,doc_id,label =data.split("\t")
		line_arr = line.split("\t")
		size = len(line_arr)
		if size != 14:
			print("err not 14")
			continue
		query_id = line_arr[0]
		query = line_arr[1]
		title = line_arr[2]
		doc_id = line_arr[12]
		label = line_arr[13]
		#print ("query_id is ->",query_id)
		#print ("query is ->",query)
		#print ("title is ->",title)
		#print ("doc_id is ->",doc_id)
		#print ("label is ->",label)
		biz_type,biz_id = doc_id.split("-")
		total_feature_dict[idx]={"title":make_feature(title,"",biz_type)}
		total_feature_dict[idx]["label"] = label
		total_feature_dict[idx]["doc_id"] = doc_id
		total_feature_dict[idx]["query_id"] = query_id
		total_feature_dict[idx]["query_str"] = query
		idx += 1
		#
		if idx%5000 == 0:
			print("cnt is:{}, {}".format(idx, idx/allsum))
			#
	f_in.close()
	time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	print("time is:", time_now)
	return total_feature_dict

len_seg_list = []
len_long_seg_list = []
len_content_seg_list = []
len_title_list = []
len_long_title_list = []
len_content_list = []
len_series_list = []
len_long_series_list = []
len_content_series_list = []
len_brand_list = []
len_long_brand_list = []
len_content_brand_list = []
len_query_list = []
len_query_seg_list = []
len_query_brand_list = []
len_query_series_list = []
def get_feature_list(total_feature_dict):
	for item in total_feature_dict:
		#原文长度特征
		len_query_list.append(total_feature_dict[item]["query"]["len_title"])
		len_title_list.append(total_feature_dict[item]["title"]["len_title"])
		len_long_title_list.append(total_feature_dict[item]["long_title"]["len_title"])
		len_content_list.append(total_feature_dict[item]["content"]["len_title"])
		#切词后长度特征
		len_query_seg_list.append(total_feature_dict[item]["query"]["len_seg"])
		len_seg_list.append(total_feature_dict[item]["title"]["len_seg"])
		len_long_seg_list.append(total_feature_dict[item]["long_title"]["len_seg"])
		len_content_seg_list.append(total_feature_dict[item]["content"]["len_seg"])
		#车系个数特征
		len_query_series_list.append(total_feature_dict[item]["query"]["len_series"])
		len_series_list.append(total_feature_dict[item]["title"]["len_series"])
		len_long_series_list.append(total_feature_dict[item]["long_title"]["len_series"])
		len_content_series_list.append(total_feature_dict[item]["content"]["len_series"])
		#品牌个数特征
		len_query_brand_list.append(total_feature_dict[item]["query"]["len_brand"])
		len_brand_list.append(total_feature_dict[item]["title"]["len_brand"])
		len_long_brand_list.append(total_feature_dict[item]["long_title"]["len_brand"])
		len_content_brand_list.append(total_feature_dict[item]["content"]["len_brand"])
		##



def normal_feature(tag, itype):
	"""
	归一化
	"""
	#return str(float(int(tag)-int(min(itype)))/float(int(max(itype))-int(min(itype))))
	return str(tag) 

def save_feature(feature_dict,yesterday,output_file):
	#fw = codecs.open("/data/home/ds/chenyongsheng/ltr/data/query_title_content_count_feature/title_count_feature." + str(yesterday), "w", "utf-8")
	fw = codecs.open(output_file, "w", "utf-8")
	#fw = codecs.open("/data/home/ds/chenyongsheng/ltr/data/base_data_for_query_title_content_feature/test_out." + str(yesterday), "w", "utf-8")
	cnt = 0
	for qid in feature_dict:
		cnt += 1
		if cnt%10000 == 0:
			print ("output_cnt is ->",cnt)
		label = feature_dict[qid]["label"].strip()
		docid = feature_dict[qid]["doc_id"].strip()
		query_id = feature_dict[qid]["query_id"].strip()
		query = feature_dict[qid]["query_str"].strip()
		#原始内容长度 特征
		#query_num = normal_feature(feature_dict[qid]["query"]["len_title"], len_query_list)
		title_num = normal_feature(feature_dict[qid]["title"]["len_title"], len_title_list)
		#content_num = normal_feature(feature_dict[qid]["content"]["len_title"], len_content_list)
		#切词后内容长度 特征
		#query_seg_num = normal_feature(feature_dict[qid]["query"]["len_seg"], len_query_seg_list)
		title_seg_num = normal_feature(feature_dict[qid]["title"]["len_seg"], len_seg_list)
		#content_seg_num = normal_feature(feature_dict[qid]["content"]["len_seg"], len_content_seg_list)
		#车系个数 特征
		#query_series_num = normal_feature(feature_dict[qid]["query"]["len_series"], len_query_series_list)
		title_series_num = normal_feature(feature_dict[qid]["title"]["len_series"], len_series_list)
		#content_series_num = normal_feature(feature_dict[qid]["content"]["len_series"], len_content_series_list)
		#品牌个数 特征
		#query_brand_num = normal_feature(feature_dict[qid]["query"]["len_brand"], len_query_brand_list)
		title_brand_num = normal_feature(feature_dict[qid]["title"]["len_brand"], len_brand_list)
		#content_brand_num = normal_feature(feature_dict[qid]["content"]["len_brand"], len_content_brand_list)
		## is_car_series
		#query_is_car_series = normal_feature(feature_dict[qid]["query"]["data_is_series"], 99999)
		title_is_car_series = normal_feature(feature_dict[qid]["title"]["data_is_series"], 99999)
		## contain_car_series
		#query_contain_car_series = normal_feature(feature_dict[qid]["query"]["data_contain_series"], 99999)
		title_contain_car_series = normal_feature(feature_dict[qid]["title"]["data_contain_series"], 99999)

		## is_car_brand
		#query_is_car_brand = normal_feature(feature_dict[qid]["query"]["data_is_brand"], 99999)
		title_is_car_brand = normal_feature(feature_dict[qid]["title"]["data_is_brand"], 99999)
		## contain_car_brand
		#query_contain_car_brand = normal_feature(feature_dict[qid]["query"]["data_contain_brand"], 99999)
		title_contain_car_brand = normal_feature(feature_dict[qid]["title"]["data_contain_brand"], 99999)

		## is_car_factory
		#query_is_factory = normal_feature(feature_dict[qid]["query"]["data_is_factory"], 99999)
		title_is_factory = normal_feature(feature_dict[qid]["title"]["data_is_factory"], 99999)
		## contain_car_factory
		#query_contain_factory = normal_feature(feature_dict[qid]["query"]["data_contain_factory"], 99999)
		title_contain_factory = normal_feature(feature_dict[qid]["title"]["data_contain_factory"], 99999)
		## count car factory 
		#query_count_car_factory = normal_feature(feature_dict[qid]["query"]["data_count_car_factory"], 99999)
		title_count_car_factory = normal_feature(feature_dict[qid]["title"]["data_count_car_factory"], 99999)

		## is_city
		#query_is_city  = normal_feature(feature_dict[qid]["query"]["data_is_city"], 99999)
		title_is_city = normal_feature(feature_dict[qid]["title"]["data_is_city"], 99999)
		## contain_city
		#query_contain_city = normal_feature(feature_dict[qid]["query"]["data_contain_city"], 99999)
		title_contain_city = normal_feature(feature_dict[qid]["title"]["data_contain_city"], 99999)

		## is_price
		#query_is_price  = normal_feature(feature_dict[qid]["query"]["data_is_price"], 99999)
		title_is_price = normal_feature(feature_dict[qid]["title"]["data_is_price"], 99999)
		## contain_price
		#query_contain_price = normal_feature(feature_dict[qid]["query"]["data_contain_price"], 99999)
		title_contain_price = normal_feature(feature_dict[qid]["title"]["data_contain_price"], 99999)

		#result = query + "\t" + query_id+"\t"+docid+"\t"+label+"\t"+query_num+"\t"+query_seg_num+"\t"+query_series_num+"\t"+ query_brand_num+"\t"+ query_is_car_series + "\t" + query_contain_car_series + "\t" + query_is_car_brand + "\t" + query_contain_car_brand + "\t" + query_is_factory + "\t" + query_contain_factory + "\t" + query_count_car_factory + "\t" + query_is_price + "\t" + query_contain_price + "\t" + query_is_city + "\t" + query_contain_city + "\t" + title_num +"\t"+ title_seg_num +"\t"+title_series_num+"\t"+title_brand_num+ "\t"+ title_is_car_series + "\t" + title_contain_car_series + "\t" + title_is_car_brand + "\t" + title_contain_car_brand + "\t" + title_is_factory + "\t" + title_contain_factory + "\t" + title_count_car_factory + "\t" + title_is_price + "\t" + title_contain_price + "\t" + title_is_city + "\t" + title_contain_city + "\t" + content_num+"\t"+content_seg_num+"\n"
		result = query + "\t" + query_id+"\t"+docid+"\t"+label+"\t"+ title_num +"\t"+ title_seg_num +"\t"+title_series_num+"\t"+title_brand_num+ "\t"+ title_is_car_series + "\t" + title_contain_car_series + "\t" + title_is_car_brand + "\t" + title_contain_car_brand + "\t" + title_is_factory + "\t" + title_contain_factory + "\t" + title_count_car_factory + "\t" + title_is_price + "\t" + title_contain_price + "\t" + title_is_city + "\t" + title_contain_city + "\n"
		#print("result is ->",result)
		fw.write(result)
	fw.close()

import datetime
def getYesterday(): 
	today=datetime.date.today()
	oneday=datetime.timedelta(days=1)
	yesterday=today-oneday
	return yesterday

if __name__=="__main__":
	yesterday = getYesterday()
	data_path = sys.argv[1]
	output_file = sys.argv[2]
	allsum = float(sys.argv[3])
	total_feature_dict = get_data(data_path, allsum)
	save_feature(total_feature_dict,yesterday,output_file)
	print("--->ok inf:{},outf:{}\n".format(data_path, output_file))
