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

if socket.gethostname() == 'iaslgpu3':
    sys.path.append('/home/neha/nlp/NeonWorkspace_1.6/ValidationSet_Generation')
elif  socket.gethostname() == 'iaslgpu5':
    sys.path.append('/home/iasl/Neha_W/NeonWorkspace_1.6/ValidationSet_Generation')

#sys.path.append('/home/iasl/Neha_W/NeonWorkspace_1.6/ValidationSet_Generation')


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

y_text = []
y_pred = []
y_gold = []
for i in range(1,int(config_desc['foldSplit'])+1):
    print("fold :",i)
    test_file = os.path.join(config_desc['resultDir'],str(i),'test.tsv')
    testdf = pd.read_csv(test_file, sep="\t", index_col=0)
    test_class = list(testdf["label"])
    test_text = list(testdf["sentence"])
    y_gold.extend(test_class)
    y_text.extend(test_text)
    
    pred_ndarray = np.array([])
    pred_file=""
    for m in range(1,int(config_desc['testFold'])+1):
        pred_file = os.path.join(config_desc['resultDir'],str(i),'test_results'+str(m)+'.tsv')
        #pred_file = os.path.join(config_desc['resultDir'],str(i),'test_results.tsv')
        preddf = pd.read_csv(pred_file, sep="\t", header=None)
        pred = [preddf.iloc[i].tolist() for i in preddf.index]
        pred_class = np.array([[np.argmax(v) for v in pred]])
        pred_class = np.transpose(pred_class)
        if pred_ndarray.shape[0] == 0:
            pred_ndarray = pred_class
        else:
            pred_ndarray = np.concatenate((pred_ndarray, pred_class), axis=1)
       
    pred_class=[]
    for index in range(pred_ndarray.shape[0]):
        decoy_dict = dict(Counter(pred_ndarray[index]))
        pred_class.append(sorted(decoy_dict.items(), key=itemgetter(1), reverse=True)[0][0])
    
    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_class)
    results = dict()
    results["f1 score"] = f[1]
    results["recall"] = r[1]
    results["precision"] = p[1]
    results["specificity"] = r[0]
    for k,v in results.items():
        print("{:11s} : {:.2%}".format(k,v))
    '''
    for i in range(len(test_class)): 
        print(pred_ndarray[i],'\t',pred_class[i],'\t',test_class[i])
    '''
    
    y_pred.extend(pred_class)
    

print('\n')
p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=y_pred, y_true=y_gold)
results = dict()
results["f1 score"] = f[1]
results["recall"] = r[1]
results["precision"] = p[1]
results["specificity"] = r[0]

a = sklearn.metrics.f1_score(y_pred=y_pred, y_true=y_gold, labels=[0,1], average='macro')  
print(a)

for k,v in results.items():
    print("{:11s} : {:.2%}".format(k,v))

fn = 0
fp = 0
addben = 0
noben = 0
for i in range(len(y_pred)):
    if (y_gold[i]==1) and (y_pred[i] == 0):
        fn += 1
        #print(i,'\t',y_pred[i],'\t',y_gold[i],'\t',y_text[i])
       
    if (y_gold[i]==0) and (y_pred[i] == 1):
        fp += 1
        a = list(filter(lambda currVal : (currVal == 'BENTITY') ,y_text[i].split()))
        if len(a) > 0:
            #print(i,'\t',y_pred[i],'\t',y_gold[i],'\t',len(a),'\t',y_text[i])
            addben = addben +1
        else:
            noben += 1
        
    
print('fn::',fn,'\t fp::',fp,'\t multiple::',addben,'\t single::',noben)


    
