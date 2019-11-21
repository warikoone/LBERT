'''
Created on Aug 23, 2019

@author: neha@iasl
'''

import os
import sys
import re
import json
import six
import tensorflow as tf
import collections
from operator import itemgetter 
from sklearn.model_selection import KFold
import numpy as np

class ValadationSet(object):
    '''
    Validation data sets for multi-class label data sets
    '''
    def __init__(self):
        self.configFileDesc = {}
        self.rawTextDict = {}
        self.rawPosDict = {}
        self.rawChunkDict = {}
        self.max = 0
    
    def writeToFile(self, passIndex, bufferArray, fileType):
        
        path = os.getcwd()#.path.dirname(sys.argv[0])
        outFilePath = re.sub("/sorting", '/processed', path)
        outFilePath = outFilePath+"/"+str(passIndex)
        if not os.path.exists(outFilePath):
            os.mkdir(outFilePath)
        outFileName = outFilePath+"/"+str(fileType)+".tsv"
        outPosFileName = outFilePath+"/"+str(fileType)+"Pos.tsv"
        outChunkFileName = outFilePath+"/"+str(fileType)+"Chunk.tsv"
        tier1FWriteBuffer = open(outFileName,'w')
        tier2FWriteBuffer = open(outPosFileName,'w')
        tier3FWriteBuffer = open(outChunkFileName,'w')
        
        if fileType == 'test':
            tier1FWriteBuffer.write('index\tsentence\tlabel\n')
            tier2FWriteBuffer.write('index\tsentence\tlabel\n')
            tier3FWriteBuffer.write('index\tsentence\tlabel\n')
        '''
        Labels for added for CPR and REL datasets 
        Alternate between the commented section for CPR and REL datasets
        '''
        for index, value in enumerate(bufferArray):
            #Original Text
            tier1BufferDict = self.rawTextDict.get(value)
            for instanceText, instanceType in tier1BufferDict.items():
                '''
                if instanceType == "1":
                    instanceType = "cpr:1"
                if instanceType == "2":
                    instanceType = "cpr:2"
                if instanceType == "3":
                    instanceType = "cpr:3"
                if instanceType == "4":
                    instanceType = "cpr:4"
                if instanceType == "5":
                    instanceType = "cpr:5"
                if instanceType == "6":
                    instanceType = "cpr:6"
                if instanceType == "7":
                    instanceType = "cpr:7"
                if instanceType == "8":
                    instanceType = "cpr:8"
                if instanceType == "9":
                    instanceType = "cpr:9"
                if instanceType == "10":
                    instanceType = "false"
                if instanceType == "-1":
                    instanceType = "false"
                '''
                if instanceType == "1":
                    instanceType = "1"
                if instanceType == "2":
                    instanceType = "2"
                if instanceType == "10":
                    instanceType = "false"
                bufferText =""
                if fileType == 'test':
                    bufferText = str(index+1)+"\t"
                bufferText = bufferText+instanceText.strip()+'\t'+instanceType+'\n' 
                tier1FWriteBuffer.write(bufferText)
                
            #Pos-Tag Text
            tier2BufferDict = self.rawPosDict.get(value)
            for instanceText, instanceType in tier2BufferDict.items():
                '''
                if instanceType == "1":
                    instanceType = "cpr:1"
                if instanceType == "2":
                    instanceType = "cpr:2"
                if instanceType == "3":
                    instanceType = "cpr:3"
                if instanceType == "4":
                    instanceType = "cpr:4"
                if instanceType == "5":
                    instanceType = "cpr:5"
                if instanceType == "6":
                    instanceType = "cpr:6"
                if instanceType == "7":
                    instanceType = "cpr:7"
                if instanceType == "8":
                    instanceType = "cpr:8"
                if instanceType == "9":
                    instanceType = "cpr:9"
                if instanceType == "10":
                    instanceType = "false"
                if instanceType == "-1":
                    instanceType = "false"
                '''
                if instanceType == "1":
                    instanceType = "1"
                if instanceType == "2":
                    instanceType = "2"
                if instanceType == "10":
                    instanceType = "false"
                bufferText =""
                if fileType == 'test':
                    bufferText = str(index+1)+"\t"
                bufferText = bufferText+instanceText.strip()+'\t'+instanceType+'\n' 
                tier2FWriteBuffer.write(bufferText)
                
            #Chunk-Tag Text
            tier3BufferDict = self.rawChunkDict.get(value)
            for instanceText, instanceType in tier3BufferDict.items():
                '''
                if instanceType == "1":
                    instanceType = "cpr:1"
                if instanceType == "2":
                    instanceType = "cpr:2"
                if instanceType == "3":
                    instanceType = "cpr:3"
                if instanceType == "4":
                    instanceType = "cpr:4"
                if instanceType == "5":
                    instanceType = "cpr:5"
                if instanceType == "6":
                    instanceType = "cpr:6"
                if instanceType == "7":
                    instanceType = "cpr:7"
                if instanceType == "8":
                    instanceType = "cpr:8"
                if instanceType == "9":
                    instanceType = "cpr:9"
                if instanceType == "10":
                    instanceType = "false"
                if instanceType == "-1":
                    instanceType = "false"
                '''
                if instanceType == "1":
                    instanceType = "1"
                if instanceType == "2":
                    instanceType = "2"
                if instanceType == "10":
                    instanceType = "false"
                bufferText =""
                if fileType == 'test':
                    bufferText = str(index+1)+"\t"
                bufferText = bufferText+instanceText.strip()+'\t'+instanceType+'\n' 
                tier3FWriteBuffer.write(bufferText)
                
        tier1FWriteBuffer.close()
        tier2FWriteBuffer.close()
        tier3FWriteBuffer.close()
        return()
        
    def check_data_imbalance(self, trainIndex):
    
        instanceDict = {}
        negative_index = list()
        positive_index = list()
        for index in trainIndex:
            class_type = list(self.rawTextDict.get(index).values())
            for item in class_type:
                if item != '10':
                    count = 0
                    if instanceDict.__contains__(item):
                        count = instanceDict.get(item)
                    count +=1
                    instanceDict.update({item:count})
                    positive_index.append(index)
                else:
                    negative_index.append(index);
                    
        sorted_d = dict(sorted(instanceDict.items(), key=itemgetter(1), reverse=True))
        data_fit_round = list(range(int(np.ceil(len(negative_index)/len(positive_index)))))
        threshold = len(positive_index)
        print(data_fit_round,'\t',threshold)
        total_pass = list()
        for iter_t in data_fit_round:
            pass_list = list(np.random.choice(len(negative_index), threshold))
            
            curr_pass = list()
            curr_index = list(map(lambda currIndex : negative_index[currIndex], pass_list))
            curr_pass.extend(curr_index)
            curr_pass.extend(positive_index)
            print(set(curr_index).intersection(set(positive_index)))
            print(iter_t,'\t',len(pass_list),'\t',len(curr_pass),'\t',len(positive_index))
            total_pass.append(curr_pass)

    
        return(total_pass)

    def crossValidationSplit(self):
        
        bufferArray = list(self.rawTextDict.keys())
        kFoldSplitStrata = KFold(n_splits = int(self.configFileDesc["foldSplit"]))
        passIndex = 1
        for trainIndex, testIndex in kFoldSplitStrata.split(bufferArray):
            #print(passIndex,'\t',trainIndex,'\t',testIndex)
            #total_pass = self.check_data_imbalance(trainIndex)
            #for sub_train_index in total_pass:
            self.writeToFile(passIndex, trainIndex, "train")
            self.writeToFile(passIndex, testIndex, "test")
            self.writeToFile(passIndex, list(), "dev")
            passIndex = passIndex+1
            #sys.exit()
        '''    
        #trainIndex = bufferArray
        testIndex = bufferArray
        #self.writeToFile(1, trainIndex, "train")
        self.writeToFile(1, testIndex, "test")
        #self.writeToFile(1, list(), "dev")
        '''
        return()

def openConfigurationFile(decoyInstance):
    
        path = os.getcwd()#os.path.dirname(sys.argv[0])
        tokenMatcher = re.search(".*ValidationSet_Generation\/", path)
        if tokenMatcher:
            configFile = tokenMatcher.group(0)
            configFile="".join([configFile,"config.json"])
        
        #with tf.gfile.GFile(configFile, "r") as reader:
        tier1BufferDict = {}
        with open(configFile, "r") as json_file:
            data = json.load(json_file)
            for (key, value) in six.iteritems(data):
                tier1BufferDict.update({key:value})
            json_file.close()
        decoyInstance.configFileDesc = tier1BufferDict
            
        return()

def readResource(decoyInstance):

    decoyInstance.rawTextDict = collections.OrderedDict()
    with open(decoyInstance.configFileDesc["corpusFile"], "r") as bufferFile:
        currentData = bufferFile.readline()
        indexCount = 0
        while len(currentData)!=0:
            tier1BufferList = list(str(currentData).split(sep='\t'))
            instanceType = tier1BufferList[0]
            instanceText = str(tier1BufferList[2]).strip()
            if decoyInstance.max < len(instanceText.split(sep=' ')):
                decoyInstance.max = len(instanceText.split(sep=' ')) 
            tier1BufferDict = {}
            tier1BufferDict.update({instanceText:instanceType})
            decoyInstance.rawTextDict.update({indexCount:tier1BufferDict})
            indexCount = indexCount+1
            currentData = bufferFile.readline()
    return()

def readPosResource(decoyInstance):

    decoyInstance.rawPosDict = collections.OrderedDict()
    with open(decoyInstance.configFileDesc["corpusPosFile"], "r") as bufferFile:
        currentData = bufferFile.readline()
        indexCount = 0
        while len(currentData)!=0:
            tier1BufferList = list(str(currentData).split(sep='\t'))
            instanceType = tier1BufferList[0]
            instanceText = str(tier1BufferList[2]).strip()
            tier1BufferDict = {}
            tier1BufferDict.update({instanceText:instanceType})
            decoyInstance.rawPosDict.update({indexCount:tier1BufferDict})
            indexCount = indexCount+1
            currentData = bufferFile.readline()
    return()

def readChunkResource(decoyInstance):

    decoyInstance.rawChunkDict = collections.OrderedDict()
    with open(decoyInstance.configFileDesc["corpusChunkFile"], "r") as bufferFile:
        currentData = bufferFile.readline()
        indexCount = 0
        while len(currentData)!=0:
            tier1BufferList = list(str(currentData).split(sep='\t'))
            instanceType = tier1BufferList[0]
            instanceText = str(tier1BufferList[2]).strip()
            tier1BufferDict = {}
            tier1BufferDict.update({instanceText:instanceType})
            decoyInstance.rawChunkDict.update({indexCount:tier1BufferDict})
            indexCount = indexCount+1
            currentData = bufferFile.readline()
    return()

def sanityTest(decoyInstance):
    
    for index in list(decoyInstance.rawTextDict.keys()):
        textDict = list(decoyInstance.rawTextDict.get(index).keys())
        posDict = list(decoyInstance.rawPosDict.get(index).keys())
        chunkDict = list(decoyInstance.rawChunkDict.get(index).keys())
        for j in range(len(textDict)):
            if(len(textDict[j].split()) and len(posDict[j].split()) and len(chunkDict[j].split())):
                continue
            else:
                print('incorrect size',index)
                break
    return()


def main(_):
    
    validationInstance = ValadationSet()
    tf.logging.set_verbosity(tf.logging.INFO)
    openConfigurationFile(validationInstance)
    readResource(validationInstance)
    readPosResource(validationInstance)
    readChunkResource(validationInstance)
    sanityTest(validationInstance)
    validationInstance.crossValidationSplit()
    print('max', validationInstance.max)
    
if __name__ == "__main__":
    tf.app.run()
    
