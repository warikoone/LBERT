'''
Created on Jun 10, 2019

@author: iasl
'''
import os
import sys
import re
import json
import six
import tensorflow as tf
import collections
from sklearn.model_selection import KFold
import socket


if socket.gethostname() == 'iaslgpu3':
    sys.path.append('/home/neha/nlp/NeonWorkspace_1.6/ValidationSet_Generation')
elif  socket.gethostname() == 'iaslgpu5':
    sys.path.append('/home/iasl/Neha_W/NeonWorkspace_1.6/ValidationSet_Generation')

class ValadationSet(object):
    
    def __init__(self):
        self.configFileDesc = {}
        self.rawTextDict = {}
        self.rawPosDict = {}
        self.max = 0
    
    def writeToFile(self, passIndex, bufferArray, fileType):
        
        path = os.path.dirname(sys.argv[0])
        outFilePath = re.sub("/sorting", '/processed', path)
        outFilePath = outFilePath+"/"+str(passIndex)
        if not os.path.exists(outFilePath):
            os.mkdir(outFilePath)
        outFileName = outFilePath+"/"+str(fileType)+".tsv"
        outPosFileName = outFilePath+"/"+str(fileType)+"Pos.tsv"
        tier1FWriteBuffer = open(outFileName,'w')
        tier2FWriteBuffer = open(outPosFileName,'w')
        if fileType == 'test':
            tier1FWriteBuffer.write('index\tsentence\tlabel\n')
            tier2FWriteBuffer.write('index\tsentence\tlabel\n')
        for index, value in enumerate(bufferArray):
            #Original Text
            tier1BufferDict = self.rawTextDict.get(value)
            for instanceText, instanceType in tier1BufferDict.items():
                if instanceType == "-1":
                    instanceType = "0"
                bufferText =""
                if fileType == 'test':
                    bufferText = str(index+1)+"\t"
                bufferText = bufferText+instanceText.strip()+'\t'+instanceType+'\n' 
                tier1FWriteBuffer.write(bufferText)
                
            #Pos-Tag Text
            tier2BufferDict = self.rawPosDict.get(value)
            for instanceText, instanceType in tier2BufferDict.items():
                if instanceType == "-1":
                    instanceType = "0"
                bufferText =""
                if fileType == 'test':
                    bufferText = str(index+1)+"\t"
                bufferText = bufferText+instanceText.strip()+'\t'+instanceType+'\n' 
                tier2FWriteBuffer.write(bufferText)
                
        tier1FWriteBuffer.close()
        tier2FWriteBuffer.close()
        return()
        
    def crossValidationSplit(self):
        
        bufferArray = list(self.rawTextDict.keys())
        kFoldSplitStrata = KFold(n_splits = int(self.configFileDesc["foldSplit"]))
        passIndex = 1
        for trainIndex, testIndex in kFoldSplitStrata.split(bufferArray):
            #print(passIndex,'\t',trainIndex,'\t',testIndex)
            self.writeToFile(passIndex, trainIndex, "train")
            self.writeToFile(passIndex, testIndex, "test")
            self.writeToFile(passIndex, list(), "dev")
            passIndex = passIndex+1
        return()

def openConfigurationFile(decoyInstance):
    
        path = os.getcwd()
        print(path)
        tokenMatcher = re.search(".*ValidationSet_Generation\/", path)
        if tokenMatcher:
            configFile = tokenMatcher.group(0)
            configFile="".join([configFile,"config.json"])
        
        #with tf.gfile.GFile(configFile, "r") as reader:
        print(configFile)
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
    #with open(decoyInstance.configFileDesc["corpusFile"], "r") as bufferFile:
    print('now',sys.getdefaultencoding())
    fp='/home/iasl/Neha_W/Complete/OriginalStream.txt'
    #fp = (''.join(fp)).encode('utf_8')
    with open(fp, "r") as bufferFile:
        currentData = bufferFile.readline()
        indexCount = 0
        while len(currentData)!=0:
            tier1BufferList = list(str(currentData).split(sep='\t'))
            instanceType = tier1BufferList[0]
            instanceId = tier1BufferList[1]
            instanceText = str(tier1BufferList[2]).strip()
            index = 0
            bufferList = list()
            for textVal in instanceText.split(sep=' '):
                if textVal == 'TRIGGERPRI':
                    index = index+1
                    textVal = 'BENTITY'+str(index)
                elif textVal == 'PROTEINT':
                    textVal = 'BENTITY'
                bufferList.append(textVal)
            
            instanceText = ' '.join(i for i in bufferList)
            tier1BufferDict = (instanceType, instanceId, instanceText)
            decoyInstance.rawTextDict.update({indexCount:tier1BufferDict})
            indexCount = indexCount+1
            currentData = bufferFile.readline()
    return()

def readPosResource(decoyInstance):

    decoyInstance.rawPosDict = collections.OrderedDict()
    fp='/home/iasl/Neha_W/Complete/PosTaggedStream.txt'
    #with open(decoyInstance.configFileDesc["corpusPosFile"], "r") as bufferFile:
    with open(fp, "r") as bufferFile:
        currentData = bufferFile.readline()
        indexCount = 0
        while len(currentData)!=0:
            tier1BufferList = list(str(currentData).split(sep='\t'))
            instanceType = tier1BufferList[0]
            instanceId = tier1BufferList[1]
            instanceText = str(tier1BufferList[2]).strip()
            index = 0
            bufferList = list()
            for textVal in instanceText.split(sep=' '):
                if textVal == 'TRIGGERPRI':
                    index = index+1
                    textVal = 'BENTITY'+str(index)
                elif textVal == 'PROTEINT':
                    textVal = 'BENTITY'
                bufferList.append(textVal)
            
            instanceText = ' '.join(i for i in bufferList)
            tier1BufferDict = (instanceType, instanceId, instanceText)
            decoyInstance.rawPosDict.update({indexCount:tier1BufferDict})
            indexCount = indexCount+1
            currentData = bufferFile.readline()
    return()

def main(_):
    
    validationInstance = ValadationSet()
    #tf.logging.set_verbosity(tf.logging.INFO)
    #openConfigurationFile(validationInstance)
    readResource(validationInstance)
    readPosResource(validationInstance)
    path = os.getcwd()
    print(path)
    outFilePath = re.sub("/sorting", '/processed', path)
    outFileName = outFilePath+"/"+"instancedOriginalStream.txt"
    outPosFileName = outFilePath+"/"+"instancePosTaggedStream.txt"
    tier1FWriteBuffer = open(outFileName,'w')
    tier2FWriteBuffer = open(outPosFileName,'w')
    for indx, indexval in validationInstance.rawTextDict.items():
        tier1BufferTup = indexval
        instanceType = "1"
        if tier1BufferTup[0] == "-1":
            instanceType = "0"
        bufferText = instanceType+"\t"+str(tier1BufferTup[1]).strip()+'\t'+str(tier1BufferTup[2]).strip()+'\n' 
        tier1FWriteBuffer.write(bufferText)
        
        tier2BufferTup = validationInstance.rawPosDict.get(indx)
        instanceType = "1"
        if tier2BufferTup[0] == "-1":
            instanceType = "0"
        bufferText = instanceType+"\t"+str(tier2BufferTup[1]).strip()+'\t'+str(tier2BufferTup[2]).strip()+'\n' 
        tier2FWriteBuffer.write(bufferText)
        
    
    tier1FWriteBuffer.close()
    tier2FWriteBuffer.close()
    
if __name__ == "__main__":
    tf.app.run()
    