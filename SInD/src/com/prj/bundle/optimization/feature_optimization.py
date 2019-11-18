'''
Created on Jul 19, 2019

@author: neha@iasl
'''

import sys
import os
import re
import json
import six
import collections
import tensorflow as tf
import socket


if socket.gethostname() == 'GPU_1':
    sys.path.append('./SInD')
elif  socket.gethostname() == 'GPU_2':
    sys.path.append('./SInD')

import src.com.prj.bundle.optimization.entropy as entropyInstance

class InputDataStream(object):
    
    def __init__(self):
        self.configDesc = {}
        self.PostTagStream = {}
        
    def featureConvolution(self, appendType, pattern_index):
        
        '''
        Instantiate feature convolution to generate localised n-frames
        '''
        
        featureInstance = entropyInstance.PatternEntropy(self.configDesc)
        featureInstance.PosDict = collections.OrderedDict()
        filterSize = int(self.configDesc.get("filterSize"))
        strideSize = int(self.configDesc.get("strideSize"))
        featureInstance.filterSize = int(filterSize)
        
        
        for tier1Key, tier1Value in self.PostTagStream.items():
            tier1BufferResultDict = {}
            for tier2Item in tier1Value:
                startIndex = 0
                endIndex = filterSize
                max_SequenceLength = len(tier2Item)
                while (startIndex < (max_SequenceLength-filterSize+1)) and (endIndex < max_SequenceLength):
                    endIndex = (startIndex+filterSize)
                    tier1BufferList = list(tier2Item[startIndex:endIndex])
                    if appendType == "Chunk":
                        for itemKey, itemValue in enumerate(tier1BufferList):
                            if(itemValue == "BENTITY1" or itemValue == "BENTITY2" or itemValue == "BENTITY"): 
                                tier1BufferList[itemKey] = 'P'+itemValue
                    if(tier1BufferList.count('#') == filterSize):
                        break
                    else:
                        patternString = ' '.join(value for value in tier1BufferList)
                        patternCount = 0
                        if tier1BufferResultDict.__contains__(patternString):
                            patternCount = tier1BufferResultDict.get(patternString)
                            
                        patternCount += 1
                        tier1BufferResultDict.update({patternString:patternCount})
                        #tf.logging.info("%s" %tier1BufferList+"\t %s" %tier2BufferResultList)
                        startIndex = startIndex+strideSize
                        
            tf.logging.info("feature depth for %d" %tier1Key+" class %d" %len(tier1BufferResultDict))
            featureInstance.PosDict.update({tier1Key:tier1BufferResultDict})
        
        pattern_index = featureInstance.calculate_entropy(appendType, pattern_index)
        
        return(pattern_index)
    
    def sequencePadding(self, maxTokenLength):
        
        '''
        Apply padding at terminals for sequence processing
        '''
        kernelSize = int(self.configDesc.get("filterSize"))
        strideSize = int(self.configDesc.get("strideSize"))
        if strideSize > kernelSize:
            tf.logging.info('Critical Error ~ sequencePadding()')
            sys.exit()
        addLength = kernelSize-strideSize
        if addLength > 0:
            maxTokenLength = maxTokenLength+addLength
            
        for tier1Key, tier1Value in self.PostTagStream.items():
            for tier2Index, tier2Item in enumerate(tier1Value):
                for tier1Index in range(maxTokenLength-len(tier2Item)):
                    tier2Item.append('#')
                tier1Value[tier2Index] = tier2Item
            self.PostTagStream.update({tier1Key:tier1Value})
        
        return()
    
    
    def readInputStream(self, appendType, pattern_index):
        
        '''
        Read POS Tag input file for feature pre-processing
        '''
        
        maxTokenLength = 0
        self.PostTagStream = {}
        self.PostTagStream = collections.OrderedDict()
        tier0BufferList = list()
        sentIndex = 0
        fileName = ""
        if appendType == "Pos":
            fileName = self.configDesc["posTagFile"]
        if appendType == "Chunk":
            fileName = self.configDesc["chunkFile"]
            
        with open(fileName, "r") as bufferFile:
            currentData = bufferFile.readline()
            while len(currentData)!=0:
                tier1BufferList = list(str(currentData).split(sep='\t'))
                instanceLabel = int(tier1BufferList[0].strip())
                if instanceLabel == -1:
                    instanceLabel = 0
                instanceText = list(str(tier1BufferList[2]).strip().rsplit(sep=' '))
                instanceList = list()
                if self.PostTagStream.__contains__(instanceLabel):
                    instanceList = self.PostTagStream.get(instanceLabel)
                instanceList.append(instanceText)
                self.PostTagStream.update({instanceLabel:instanceList})
                tier0BufferList.extend(instanceText)
                if maxTokenLength < len(instanceText):
                    maxTokenLength = len(instanceText)
                
                sentIndex +=1
                
                '''
                if sentIndex == 10:
                    break
                '''
                currentData = bufferFile.readline()
        bufferFile.close()

        self.sequencePadding(maxTokenLength)
        pattern_index = self.featureConvolution(appendType, pattern_index)        
        return(pattern_index)

def openConfigurationFile(decoyInstance):
    
    '''
    Load json configuration file
    '''
    
    path = os.getcwd()
    tokenMatcher = re.search(".*SInD\/", path)
    configFile = ""
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
    decoyInstance.configDesc = tier1BufferDict
            
    return()


def main(_):
    
    pattern_index = 0
    inputStreamInstance = InputDataStream()
    tf.logging.set_verbosity(tf.logging.INFO)
    openConfigurationFile(inputStreamInstance)
    pattern_index = inputStreamInstance.readInputStream("Pos", pattern_index)
    #pattern_index = inputStreamInstance.readInputStream("Chunk", pattern_index)
    
if __name__ == "__main__":
    tf.app.run()
