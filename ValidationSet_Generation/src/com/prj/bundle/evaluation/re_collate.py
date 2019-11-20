'''
Created on Aug 6, 2019

@author: iasl
'''
import os,sys
import numpy as np
import pandas as pd
import sklearn.metrics
import argparse
import json
import re
import six
from collections import Counter
from operator import itemgetter
import socket
import path

if socket.gethostname() == 'iaslgpu3':
    sys.path.append('/home/neha/nlp/NeonWorkspace_1.7/ValidationSet_Generation')
elif  socket.gethostname() == 'iaslgpu5':
    sys.path.append('/home/iasl/Neha_W/NeonWorkspace_1.7/ValidationSet_Generation')

#sys.path.append('/home/iasl/Neha_W/NeonWorkspace_1.6/ValidationSet_Generation')


def reverseMap():
    key_map = {}
    key_map.update({0:0})
    key_map.update({1:1})
    key_map.update({2:2})
    key_map.update({3:3})
    key_map.update({4:4})
    key_map.update({5:5})
    key_map.update({6:0.5})
    key_map.update({7:1.5})
    key_map.update({8:2.5})
    key_map.update({9:3.5})
    key_map.update({10:4.5})
    
    return(key_map)

def openConfigurationFile(config_desc):
    
    path = os.getcwd()
    tokenMatcher = re.search(".*ValidationSet_Generation\/", path)
    configFile = ""
    if tokenMatcher:
        configFile = tokenMatcher.group(0)
        configFile="".join([configFile,"config.json"])
        
    #with tf.gfile.GFile(configFile, "r") as reader:
    with open(configFile, "r") as json_file:
        data = json.load(json_file)
        for (key, value) in six.iteritems(data):
            config_desc.update({key:value})
        json_file.close()
           
    return(config_desc)

config_desc = {}
config_desc = openConfigurationFile(config_desc)

y_pred = []
y_gold = []
col_pred_ndarray = np.array([])
for i in range(1,int(config_desc['foldSplit'])+1):
    print("fold :",i)
    test_file = os.path.join(config_desc['resultDir'],str(i),'test.tsv')
    testdf = pd.read_csv(test_file, sep="\t", index_col=0)
    str_to_int_mapper = dict()
    for j,v in enumerate(sorted(testdf["label"].unique())):
        str_to_int_mapper[v] = j
    test_class = [str_to_int_mapper[v] for v in testdf["label"]]
    y_gold.extend(test_class)
    
    pred_ndarray = np.array([])
    pred_file=""
    for m in range(1,int(config_desc['testFold'])+1):
        pred_file = os.path.join(config_desc['resultDir'],str(i),'test_results'+str(m)+'.tsv')
        preddf = pd.read_csv(pred_file, sep="\t", header=None)
        pred = [preddf.iloc[i].tolist() for i in preddf.index]
        pred_class = [np.argmax(v) for v in pred]
    y_pred.extend(pred_class)
    
    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_class, labels=[0,1,2,3,4], average="micro")
    results = dict()
    results["f1 score"] = f
    results["recall"] = r
    results["precision"] = p
    for k,v in results.items():
        print("{:11s} : {:.2%}".format(k,v))
        
            
print('\n')
p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=y_pred, y_true=y_gold, labels=[0,1,2,3,4], average="micro")
print(sklearn.metrics.classification_report(y_pred=y_pred, y_true=y_gold, digits=3))
results = dict()
results["f1 score"] = f
results["recall"] = r
results["precision"] = p
for k,v in results.items():
    print("{:11s} : {:.2%}".format(k,v))
    
'''
path = os.getcwd()
outFilePath = os.path.join(path,'score.txt')
tier1FWriteBuffer = open(outFilePath,'w')
key_map = reverseMap()
for i in y_pred:
    value = -1
    if key_map.__contains__(i):
        value = key_map[i]
    else:
        print('error', i)
        sys.exit()
    tier1FWriteBuffer.write(str(value)+'\n')

tier1FWriteBuffer.close()
'''
        
