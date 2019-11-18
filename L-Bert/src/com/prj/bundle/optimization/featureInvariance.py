'''
Created on Jun 12, 2019

@author: nehawarikoo
'''

import sys
import re
import os
import operator
import six
import collections
import math
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from tensorflow import set_random_seed
from numpy.random import seed
from decimal import Decimal

class SemanticInvariance(object):
    
    def __init__(self, embed_locale, feature_locale):
        
        self.pretrained_Embedding = np.array([])#KeyedVectors.load_word2vec_format(embed_locale,binary=True)
        self.feature_map = {}
        self.embeddingDimension = 300
        self.unseenEmbedding = {}
        self.sequence_map={}
        self.posPattern_map={}
        seed(1)
        set_random_seed(2)
        
        '''
        if(len(self.pretrained_Embedding.vocab)) == 0:
            tf.logging.info("Error loading the pre-trained embedding")
        '''
        
        #self.assemble_feature_map(feature_locale)
        #self.load_sequence_map(sequence_locale)
        self.load_posPattern_map(feature_locale)
        
        tf.logging.info('Loading Semantic Invariance module')
    
    
    def load_posPattern_map(self, feature_locale):
        
        ''' load POS context n-frames from the Universal Lexical Cluster Map'''
        
        self.posPattern_map = collections.OrderedDict()
        with open(os.path.join(feature_locale,'feature.tsv'), "r") as bufferReader:
            currentLine = bufferReader.readline()
            while len(currentLine)!=0:
                currentLine = currentLine.strip()
                tier1BufferList = list(currentLine.split('\t'))
                pattern_label = np.int32(tier1BufferList[0])
                pattern_list = list(str(tier1BufferList[1]).split(', '))
                self.posPattern_map.update({pattern_label:pattern_list})
                currentLine = bufferReader.readline()
                
        tf.logging.info('pos pattern map length : %d'%len(self.posPattern_map))
        bufferReader.close()
        
        return()
    
    def load_sequence_map(self, sequence_locale):
        
        with open(sequence_locale, "r") as bufferReader:
            currentLine = bufferReader.readline()
            while len(currentLine)!=0:
                tier1BufferList = list(currentLine.split('\t'))
                label = np.int32(tier1BufferList[0])
                sequence = str(tier1BufferList[1])
                score = np.float32(tier1BufferList[2])
                tier1BufferDict = {}
                if self.sequence_map.__contains__(sequence):
                    tier1BufferDict = self.sequence_map.get(sequence)
                tier1BufferDict.update({label:score})
                self.sequence_map.update({sequence:tier1BufferDict})
                currentLine = bufferReader.readline()
        tf.logging.info('sequence map length : %d'%len(self.sequence_map))
        bufferReader.close()

        return()
    
    def __semanticInvariance__(self, Pos_Sequence):
        
        ''' Calculate invariance among n-frames'''
        
        def get_components(pos_pattern):
            
            tier1BufferDict = {}
            if self.sequence_map.__contains__(pos_pattern):
                tier1BufferDict = self.sequence_map.get(pos_pattern)
            
            ts_p20 = np.float32(0)
            ts_p11 = np.float32(0)
            ts_p02 = np.float32(0)
            
            if tier1BufferDict.__contains__(0):
                ts_p20 = tier1BufferDict.get(0)
                
            if tier1BufferDict.__contains__(1):
                ts_p02 = tier1BufferDict.get(1)
            
            ts_p11 = (ts_p20*ts_p02)
            
            return(ts_p20, ts_p11, ts_p02)
        
        def __invaraintScore__(ts_p20, ts_p11, ts_p02):
            return((math.pow(ts_p20, 2))+(math.pow(ts_p11, 2)/2)+(math.pow(ts_p02, 2)))
        
        pos_pattern = ' '.join(val for val in Pos_Sequence)
        ts_p20, ts_p11, ts_p02 = get_components(pos_pattern)
        ts_invar = __invaraintScore__(ts_p20, ts_p11, ts_p02)
        
        return(ts_invar)
    
    '''
    @__geometricInvariance__() Depreciated module in this version 
    '''
    def __geometricInvariance__(self, Pos_Sequence, Lex_Sequence):
        
        def __invaraintScore__(ts_p20, ts_p11, ts_p02):
            return((math.pow(ts_p20, 2))+(math.pow(ts_p11, 2)/2)+(math.pow(ts_p02, 2)))
    
        def recursiveTokenIdentification(currentToken, remainderToken, wordSubTokens):
        
            startIndex = 0
            endIndex = len(currentToken)
            termIndex = endIndex
            bufferToken = currentToken[startIndex:termIndex]
                
            flag = 0
            if bufferToken in self.pretrained_Embedding.vocab:
                dicIndex = len(wordSubTokens)
                wordSubTokens.update({dicIndex:{1:bufferToken}})
                flag=1
                    
            if ((flag == 0) and (termIndex > 1)):
                ''' reducing one letter at a time'''
                remainderToken.append(currentToken[termIndex-1:])
                currentToken = bufferToken[:termIndex-1]
            elif(flag == 1):
                ''' subgroup word structure'''
                if len(remainderToken) > 0:
                    remainderToken.reverse()
                    currentToken = ''.join(charTerm for charTerm in remainderToken)
                    remainderToken = list()
                else:
                    currentToken = None
            else:
                ''' for single words not present with embedding'''
                dicIndex = len(wordSubTokens)
                wordSubTokens.update({dicIndex:{-1:bufferToken}})
                if len(remainderToken) > 0:
                    remainderToken.reverse()
                    currentToken = ''.join(charTerm for charTerm in remainderToken)
                    remainderToken = list()
                else:
                    currentToken = None
                
            if currentToken is not None:
                recursiveTokenIdentification(currentToken, remainderToken, wordSubTokens)
            
            return(wordSubTokens)
        
        def generate_Embedding(tokens, compositeWord):
            #randValue = tf.random_normal(shape=(1,self.embeddingDimension), stddev=0.01)
            #retValue = self.retriveSessionValue(randValue)
            assembledEmbed = np.ones([1, self.embeddingDimension], dtype = np.float32)
            for token in tokens:
                if token in self.pretrained_Embedding.vocab:
                    randValue = np.array([self.pretrained_Embedding.word_vec(token)[0:self.embeddingDimension]])
                    np.reshape(randValue, (1, self.embeddingDimension))
                else:
                    randValue = np.random.rand(1,self.embeddingDimension)
                
                assembledEmbed = np.multiply(assembledEmbed, randValue)
            
            self.unseenEmbedding.update({compositeWord:assembledEmbed})
            
            return()
        
        def retrieve_Embedding(token):
            
            if token in self.pretrained_Embedding.vocab:
                return self.pretrained_Embedding.word_vec(token)[0:self.embeddingDimension]
            else:
                if token not in self.unseenEmbedding:
                    #tf.logging.info(" Tag: %s \n"%token)
                    wordSubTokens = {}
                    wordSubTokens = recursiveTokenIdentification(token,list(), wordSubTokens)
                    generate_Embedding(wordSubTokens, token)
                return self.unseenEmbedding.get(token)[0:self.embeddingDimension]
            
        def generate_components(Pos_Sequence, Lex_Sequence):
            
            contextPosVector = np.ones([1, self.embeddingDimension], dtype=np.float32)
            contextLexVector = np.ones([1, self.embeddingDimension], dtype=np.float32)
            for indexKey, posValue in enumerate(Pos_Sequence):
                lexValue = Lex_Sequence[indexKey]
                tier1NdMatrix = np.array(retrieve_Embedding(posValue), dtype=np.float32).reshape((1, self.embeddingDimension))
                contextPosVector = np.multiply(contextPosVector, tier1NdMatrix)
                
                tier2NdMatrix = np.array(retrieve_Embedding(lexValue), dtype=np.float32).reshape((1, self.embeddingDimension))
                contextLexVector = np.multiply(contextLexVector, tier2NdMatrix)
            
            tensor_initialize = list()
            tensor_initialize.append(np.matmul(contextPosVector, np.reshape(contextPosVector, (self.embeddingDimension, 1))))
            tensor_initialize.append(np.matmul(contextPosVector, np.reshape(contextLexVector, (self.embeddingDimension, 1))))
            tensor_initialize.append(np.matmul(contextLexVector, np.reshape(contextLexVector, (self.embeddingDimension, 1))))
            
            return(tensor_initialize)
        
        def compute_invariance(tensorList):
            
            ts_p20 = tensorList[0][0,0]
            ts_p11 = tensorList[1][0,0]
            ts_p02 = tensorList[2][0,0]
            invar_Score = __invaraintScore__(ts_p20, ts_p11, ts_p02)
            #tf.logging.info("(%s)x2" %ts_p20+"+(%s)xy"%ts_p11+"+(%s)y2"%ts_p02+" IScore: %s"%invar_Score)
            
            return(invar_Score)
        
        tensor_initialize = generate_components(Pos_Sequence, Lex_Sequence)
        invar_Score = compute_invariance(tensor_initialize)
        
        return(invar_Score)
    
    '''
    @openConfigurationFile() Depreciated module in this version 
    '''
    def openConfigurationFile(self, feature_locale):
        
        ''' open feature configuration file'''
    
        def format_pattern(bufferString, pattern_list):
            pattern_list.append(list(re.findall('\w+',''.join(word for word in bufferString))))
            return(pattern_list)

        tier1BufferDict = {}
        cluster_tag = ''            
        with open(feature_locale, "r") as bufferReader:
            currentLine = bufferReader.readline()
            while len(currentLine)!=0:
                tier1BufferList = list(currentLine.split('\t'))
                if tier1BufferList[0] != 'clusterTag':
                    if len(tier1BufferList) == 4:
                        if cluster_tag != '':
                            self.feature_map.update({cluster_tag:tier1BufferDict})
                        tier1BufferDict = {}
                        cluster_tag = str(tier1BufferList[0])
                        tier1BufferList = tier1BufferList[1:4]
                    subClusterId = int(tier1BufferList[0])
                    invar_range = list()
                    match_pattern = re.compile('\\[|\\]|\\\'')
                    for entry in str(tier1BufferList[1]).split(sep=','):
                        invar_range.append(float(match_pattern.sub('',entry)))

                    pattern_list = list()
                    decoyString = tier1BufferList[2][1:len(tier1BufferList[2])-1]
                    match_pattern = re.compile('\\[|\\]')
                    matchCount=0
                    matchList = list()
                    for pattern in match_pattern.finditer(decoyString):
                        if matchCount == 2:
                            pattern_list = format_pattern(
                                decoyString[matchList[0]+1:matchList[1]], pattern_list)
                            matchCount = 0
                            matchList = list()
                        matchList.append(pattern.span()[0])
                        matchCount = matchCount +1
                    if len(matchList) > 0:
                        pattern_list = format_pattern(
                            decoyString[matchList[0]+1:matchList[1]], pattern_list)
                        
                    #tf.logging.info(subClusterId,'\t',invar_range,'\t',pattern_list)
                    tier1BufferDict.update({subClusterId:(invar_range, pattern_list)})
                currentLine = str(bufferReader.readline()).strip()
                
            if cluster_tag != '':
                self.feature_map.update({cluster_tag:tier1BufferDict})
        
        bufferReader.close()
        
        
        return()
    
    def assemble_feature_map(self, feature_locale):
        
        self.openConfigurationFile(feature_locale)
        
        return()
 
    '''
    @map_to_feature() Depreciated module in this version 
    '''
    def map_to_feature(self, cluster_tag, pos_pattern, invariance_score):
        
        def minimal_variance_cluster(source_tuple, target_Score):
            
            tier1BufferDict = {}
            for source_set in source_tuple:
                invar_diff = abs(source_set[1]-target_Score)
                '''
                invar_diff = (source_set[1]/target_Score)
                if (source_set[1] > target_Score):
                    invar_diff = (target_Score/source_set[1])
                '''
                tier1BufferDict.update({invar_diff:source_set[0]})
            
            tier1BufferDict = sorted(tier1BufferDict.items(), key=operator.itemgetter(0))
            
            #print(tier1BufferDict[0][1])
            return(tier1BufferDict[0][1])
        
        def initiate_pattern_match(source_list, target):
            
            for source in source_list:
                if len(source) == len(target):
                    ref_score = round(len(source)/2)
                    target_score = 0
                    for index in range(len(source)):
                        if source[index] == target[index]:
                            target_score = target_score+1
                    if target_score >= ref_score:
                        return (True, target_score)
            return (False, -1)
        
        def isolate_relevant_items(bufferDict, pos_pattern, invariance_score):
            
            target_range = {}
            for (itemKey, itemValue) in six.iteritems(bufferDict):
                invar_range = itemValue[0]
                pattern_list = itemValue[1]
                retVal, retScore = initiate_pattern_match(pattern_list, pos_pattern)
                if retVal:
                    target_list = list()
                    if target_range.__contains__(retScore):
                        target_list = target_range.get(retScore)
                    target_list.append((itemKey, invar_range[0]))
                    target_range.update({retScore:target_list})

            if len(target_range) == 0:
                target_list = list()
                for (itemKey, itemValue) in six.iteritems(bufferDict):
                    invar_range = itemValue[0]
                    target_list.append((itemKey, invar_range[0]))
                target_range.update({1:target_list})
            
            target_range = sorted(target_range.items(), key=operator.itemgetter(0), 
                                       reverse=True)
            target_range = target_range[0]

            return(minimal_variance_cluster(target_range[1], invariance_score))
        
        def get_parentCluster_index(cluster_tag):
            
            target_clusterId = 9
            for (itemKey,itemValue) in six.iteritems(self.feature_map):
                #print(target_clusterId,'\t',len(itemValue),'\t',itemKey)
                if itemKey == cluster_tag:
                    break
                else:
                    target_clusterId = target_clusterId+len(itemValue)

            return(target_clusterId)
        
        target_clusterId = -1
        if self.feature_map.__contains__(cluster_tag):
            group_clusterId = get_parentCluster_index(cluster_tag)
            tier1BufferDict = self.feature_map.get(cluster_tag)
            target_clusterId = group_clusterId + isolate_relevant_items(
                tier1BufferDict, pos_pattern, invariance_score)
        else:
            tf.logging.info('isolate_relevant_items() ~ Unidentified cluster_tag')
            
        
        if target_clusterId >= 16502:
            tf.logging.info('%s' %cluster_tag +'\t %s' %pos_pattern+'\t %s' %target_clusterId)
        
        
        return(target_clusterId, group_clusterId+1)
    
    
    def map_context_to_pattern(self, context_sequence, pattern_type):
        
         ''' Map POS context n-frame with the pre-trained n-frame features '''
            
        def normalize(sequence, pattern_type):
            
            for key,value in enumerate(sequence):
                if value == 'BENTITY1' or value == 'BENTITY2' or value == 'BENTITY':
                    if pattern_type == "pos":
                        sequence[key] = 'NN'
                    if pattern_type == "chunk":
                        sequence[key] = 'NP' 
            
            return(sequence)
        
        def partial_pattern_match(target_sequence, lead_cluster_id, 
                                  universal_pattern_map, pattern_type):

            ''' partial match between n-frames'''
            
            '''
            Args:
                target_sequence = sequence which is to be compared
                lead_cluster_id = best cluster id matches from ULC based on the lead sequence in the frame
                universal_pattern_map = pre-trained cluster id map 
                pattern_type = "chunk/POS-tag"
            '''
            ref_score = round(len(target_sequence)/2)            
            pattern_id_list = list(universal_pattern_map.keys())
            start_index = pattern_id_list.index(lead_cluster_id)
            pattern_id_list = pattern_id_list[start_index:len(universal_pattern_map)]
            for pattern_index in pattern_id_list:
                source_list = universal_pattern_map.get(pattern_index)
                for current_sequence in source_list:
                    source_sequence = list(current_sequence.split(' '))
                    if len(source_sequence) == len(target_sequence):
                        target_score = 0
                        for index in range(len(source_sequence)):
                            if source_sequence[index] == target_sequence[index]:
                                target_score = target_score+1
                            if target_score >= ref_score:
                                return (pattern_index)
            
            ''' optimize with normalized tags'''
            target_sequence = normalize(target_sequence, pattern_type)
            for pattern_index in pattern_id_list:
                source_list = universal_pattern_map.get(pattern_index)
                for current_sequence in source_list:
                    source_sequence = list(current_sequence.split(' '))
                    if len(source_sequence) == len(target_sequence):
                        target_score = 0
                        source_sequence = normalize(source_sequence, pattern_type)
                        for index in range(len(source_sequence)):
                            if source_sequence[index] == target_sequence[index]:
                                target_score = target_score+1
                            if target_score >= ref_score:
                                return (pattern_index)
            
            return(-1)
        
        if pattern_type == "chunk":
            for itemKey, itemValue in enumerate(context_sequence):
                if(itemValue == "BENTITY1" or itemValue == "BENTITY2" or itemValue == "BENTITY"): 
                    context_sequence[itemKey] = 'P'+itemValue

        lead_tag = context_sequence[0]
        lead_cluster_id = -1
        complete_context_id = -1
        context_sequence_string = ' '.join(i for i in context_sequence)
        universal_pattern_map = self.posPattern_map
            
        for (index, value) in six.iteritems(universal_pattern_map):
            pattern_match = list(filter(lambda currVal : currVal 
                                        == context_sequence_string, value))
            cluster_match = list(filter(lambda currVal : currVal 
                                        == lead_tag, value))
            if len(cluster_match) > 0:
                lead_cluster_id = index
                
            if len(pattern_match) > 0:
                complete_context_id = index
                break
        
        if lead_cluster_id != -1 and complete_context_id == -1:
            complete_context_id = partial_pattern_match(context_sequence, lead_cluster_id, 
                                                        universal_pattern_map, pattern_type)
        '''
        In case of no match, append default tag id
        '''
        if lead_cluster_id == -1 or complete_context_id == -1:
            tf.logging.info('critical error map_context_to_pattern() ~ Unidentified cluster_tag')
            tf.logging.info('%s'%context_sequence)
            lead_cluster_id = 1
            complete_context_id = 2
            
        return(lead_cluster_id, complete_context_id)
    
