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

class ValadationSet(object):
    
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
        for index, value in enumerate(bufferArray):
            #Original Text
            tier1BufferDict = self.rawTextDict.get(value)
            for instanceText, instanceType in tier1BufferDict.items():
                if instanceType == "-1":
                    instanceType = "0"
                else:
                    instanceType = "1"
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
                else:
                    instanceType = "1"
                bufferText =""
                if fileType == 'test':
                    bufferText = str(index+1)+"\t"
                bufferText = bufferText+instanceText.strip()+'\t'+instanceType+'\n' 
                tier2FWriteBuffer.write(bufferText)
                
            #Chunk-Tag Text
            tier3BufferDict = self.rawChunkDict.get(value)
            for instanceText, instanceType in tier3BufferDict.items():
                if instanceType == "-1":
                    instanceType = "0"
                else:
                    instanceType = "1"
                bufferText =""
                if fileType == 'test':
                    bufferText = str(index+1)+"\t"
                bufferText = bufferText+instanceText.strip()+'\t'+instanceType+'\n' 
                tier3FWriteBuffer.write(bufferText)
                
        tier1FWriteBuffer.close()
        tier2FWriteBuffer.close()
        tier3FWriteBuffer.close()
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

def main(_):
    
    validationInstance = ValadationSet()
    tf.logging.set_verbosity(tf.logging.INFO)
    openConfigurationFile(validationInstance)
    readResource(validationInstance)
    readPosResource(validationInstance)
    readChunkResource(validationInstance)
    validationInstance.crossValidationSplit()
    print('max', validationInstance.max)
    
if __name__ == "__main__":
    tf.app.run()
    