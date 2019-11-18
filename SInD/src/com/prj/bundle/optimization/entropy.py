'''
Created on Jul 19, 2019

@author: neha@iasl
'''

import sys
import re
import operator
import six
import collections
import math
import tensorflow as tf
import numpy as np
from _operator import itemgetter
import os

class PatternEntropy(object):
    
    def __init__(self, configDesc):
        
        self.featurePath = configDesc['featureFile']
        self.threshold = float(configDesc['invarThreshold'])
        self.filterSize = 0
        self.PosDict = {}
        self.feature_matrix = {}
        self.contextDimension = 768
    
    def find_pattern_match(self, startIndex, endIndex, lead_list, complete_pattern_list):
        
        #print('lead string', lead_list)
        tier1BufferList = list(filter(
            lambda currList : currList[startIndex:endIndex] == lead_list, complete_pattern_list))
        
        tier1BufferResultList = list(map(lambda currVal: ' '.join(val for val in currVal), 
                                         tier1BufferList))
        #print(tier1BufferResultList)

        return(tier1BufferResultList)
    
    def pattern_count(self, pattern, decoyDict):
        
        pattern_list = []
        score_count = 0
        if isinstance(pattern, str):
            pattern_list.append(pattern)
        elif isinstance(pattern, list):
            pattern_list = pattern
            
        for lead_pattern in pattern_list:
            for (key, value) in six.iteritems(decoyDict):
                if key == lead_pattern:
                    score_count = score_count+value
                
        score_count = np.float32(score_count)
        return(score_count)
    
    def calculate_probability(self, current_pattern_list, complete_pattern_list, decoyDict):
        
        entropy_score = np.float32(0)
        index_list = list(range(self.filterSize))
        index_list.reverse()
        for i in index_list:
            prob_score = np.float32(1)
            if i > 0:
                lead_list = current_pattern_list[0:i]
                secondary_list = self.find_pattern_match(0, i, lead_list, complete_pattern_list)
                num_score = self.pattern_count(' '.join(val for val in current_pattern_list), decoyDict)
                den_score = self.pattern_count(secondary_list, decoyDict)
                if den_score > 0:
                    prob_score = (num_score/den_score)
            log_value = ((prob_score*np.log(prob_score))*np.float32(-1))
            entropy_score = entropy_score + log_value
            #print(i,'\t',log_value)

        #print(current_pattern_list,'\t',entropy_score)
        return(entropy_score)
    
    def writeToFeatureFile(self, label, pattern_list, score, append_status):
            
            tier1BufferWriter = open(self.featurePath,append_status)
            #tier1BufferWriter.write('clusterTag\tclusterId\tScore\tPattern\n')
            currentWritePointer = str(label)+"\t"+str(
                ' '.join(val for val in pattern_list)+"\t"+str(score)+"\n")
            tier1BufferWriter.write(currentWritePointer)
            tier1BufferWriter.flush()
            tier1BufferWriter.close()
                        
            return()
    
    def generate_feature_matrix(self, buffer_feature_matrix, label, pattern_list, score):
        
        lead_tag = pattern_list[0]
        pattern_text = ' '.join(val for val in pattern_list)
        tier1BufferDict = {}
        if lead_tag in buffer_feature_matrix.keys():
            tier1BufferDict = buffer_feature_matrix.get(lead_tag)
            
        tier2BufferDict = {}
        if tier1BufferDict.__contains__(pattern_text):
            tier2BufferDict = tier1BufferDict.get(pattern_text)
        tier2BufferDict.update({label:score})
        tier1BufferDict.update({pattern_text : tier2BufferDict})
        buffer_feature_matrix.update({lead_tag:tier1BufferDict})
        
        return(buffer_feature_matrix)
    
    
    
    def __invariant_optimization__(self, buffer_feature_matrix, appendType, pattern_index):
        
        '''
        Evaluate n-frames for invariance. Using algebraic optimization, 
        cluster features to generate Universal Lexical Clusters (ULC)
        '''
    
        def get_components(tier1BufferDict):
            
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
        
        def optimization(decoy_list):
            
            feature_dict = {}
            i=0
            j = i+1
            while (i < (len(decoy_list)-1)) and (j < (len(decoy_list))):
                num = decoy_list[i]
                den = decoy_list[j]
                num_val = float(num[0])#float("{:.7f}".format(num[0]))
                den_val = float(den[0])#float("{:.7f}".format(den[0]))
                if float(0.0) == float(den_val):
                    den_val = 1.0
                invar_fraction = (num_val/den_val)
                invar_fraction = float("{:.1f}".format(invar_fraction))
                #print(num,'\t',den,'\t',invar_fraction,'\t',float("{:.2f}".format(invar_fraction)),'\t',self.threshold)
                update_list = []
                if feature_dict.__contains__(num[0]):
                    update_list = feature_dict.get(num[0])
                if invar_fraction == self.threshold:
                    if not update_list.__contains__(num[1]):
                        update_list.extend(num[1])
                    if not update_list.__contains__(den[1]):
                        update_list.extend(den[1])
                    feature_dict.update({num[0]:update_list})
                else:
                    if not update_list.__contains__(num[1]):
                        update_list.extend(num[1])
                    feature_dict.update({num[0]:update_list})
                    i = j
                
                j = j+1
            
            if i == (len(decoy_list)-1):
                temp = decoy_list[i]
                feature_dict.update({temp[0]:temp[1]})
                
            
            return(feature_dict)
        
        def write_to_featurefile(lead_feature, pattern_index, invar_distribution, append_status, appendType):
            
            '''
            Write cluster embedding vectors on file
            '''
            
            string_feature = ''
            if isinstance(lead_feature, list):
                string_feature = ', '.join(i for i in lead_feature)
            else:
                string_feature = lead_feature
                
            embed_feature = ', '.join(str(i) for i in invar_distribution)

            embedding_file_name = "embedding.tsv"
            tier1BufferWriter = open(os.path.join(self.featurePath,embedding_file_name), append_status)
            currentWritePointer = str(pattern_index)+"\t"+str(embed_feature)+"\n"
            tier1BufferWriter.write(currentWritePointer)
            tier1BufferWriter.flush()
            tier1BufferWriter.close()
            
            feature_file_name = "feature.tsv"
            tier2BufferWriter = open(os.path.join(self.featurePath,feature_file_name), append_status)
            currentWritePointer = str(pattern_index)+"\t"+str(string_feature)+"\n"
            tier2BufferWriter.write(currentWritePointer)
            tier2BufferWriter.flush()
            tier2BufferWriter.close()
            
            pattern_index+=1
            
            return(pattern_index)
        
        def angular_transformation(cosine_score):
            
            angular_displacement = np.arccos([cosine_score])[0]
            print(cosine_score,'\t',np.arccos([cosine_score]))
            sine_score = np.sin(angular_displacement*np.pi/180)
            base_matrix = np.ones([self.contextDimension, self.contextDimension], dtype=np.float32)
            base_matrix = np.diag(np.diag(base_matrix))
            for i in range(0,2):
                for j in range(0,2):
                    if i == j:
                        base_matrix[i,j]=cosine_score
                    else:
                        base_matrix[i,j]=np.power(-1,i)*sine_score
            
            return(base_matrix)
        
        def generate_feature_map(feature_dict, pattern_index, lead_feature, append_status, 
                                 appendType, weight_index):
            
            init_range = list(feature_dict.keys())
            start_range = 0.0
            end_range = init_range[len(init_range)-1]
            if len(init_range) > 1:
                start_range = init_range[0]
            else:
                start_range = (end_range*1.1)
                
            tf.logging.info('%s' %pattern_index +'\t %s'%appendType+'\t %s'%start_range+'\t %s'%end_range)

            if (start_range == 0.0):
                invar_distribution = np.random.randn(
                    1, self.contextDimension).reshape(self.contextDimension, )
            else:            
                invar_distribution = np.random.uniform(start_range, end_range, self.contextDimension)
                
            pattern_index = write_to_featurefile(lead_feature, pattern_index, 
                                                 invar_distribution, append_status, appendType)
            
            lead_distribution = list(invar_distribution)
            start_range = 0.0
            for i in range(0, len(init_range)):
                end_range = init_range[i]
                if i == 0:
                    start_range = (end_range*1.1)
                
                if (start_range == 0.0):
                    invar_distribution = np.random.randn(
                        1, self.contextDimension).reshape(self.contextDimension, )
                else:
                    invar_distribution = np.random.uniform(
                        float(start_range), float(end_range), self.contextDimension)
                    
                lead_feature = feature_dict.get(end_range)
                pattern_index = write_to_featurefile(lead_feature, pattern_index, 
                                                     invar_distribution, append_status, appendType)
                start_range = end_range

            '''
            for i in range(0, len(init_range)):
                end_range = init_range[i]
                invar_distribution = np.random.uniform(
                    float(init_range[0]), float(init_range[1]), self.contextDimension)
                lead_feature = feature_dict.get(end_range)
                if i == 0:
                    pattern_index = write_to_featurefile(lead_feature, pattern_index, 
                        invar_distribution, append_status)
                else: 
                    start_range = init_range[i-1]
                    start_score = float(start_range)#float("{:.7f}".format(start_range))
                    end_score = float(end_range)#float("{:.7f}".format(end_range))
                    if float(0.0) == float(end_score):
                        end_score = 1.00
                    invar_fraction = (start_score/end_score)
                    if float(invar_fraction) == 0.0:
                        invar_fraction = 1.00

                    invar_fraction = float("{:.1f}".format(invar_fraction))
                    if invar_fraction > float(1.0):
                        invar_diff = invar_fraction-float(1.0)
                    else:
                        invar_diff = float(1.0)-invar_fraction

                    variant_distribution = list(lead_distribution)
                    if invar_diff < float(1.0):
                        #base_matrix = angular_transformation(invar_diff)
                        #invar_distribution = np.matmul(base_matrix, lead_distribution)
                        index = np.random.choice(self.contextDimension, 1)[0]
                        value = np.random.random_sample()
                        variant_distribution[index] = value
                        print(index,'\t',value)
                    #print(start_range,'\t',end_range,'\t',invar_fraction,'\t',invar_diff)
                    
                    pattern_index = write_to_featurefile(lead_feature, pattern_index, 
                                                         variant_distribution, append_status)
            '''

            return(pattern_index)

        append_status = "a+"
        if pattern_index == 0:
            append_status = "w+"
            # dummy mask features
            unused_list = ['unused0', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5', 'unused6', 
                           '[CLS]', '[SEP]', '[MASK]']
            for lead_feature in unused_list:
                invar_distribution = np.random.randn(1, self.contextDimension).reshape(self.contextDimension, )
                pattern_index = write_to_featurefile(lead_feature, pattern_index, 
                                                         invar_distribution, append_status, appendType)
                append_status = "a+"
            
        weight_index=1.0
        append_status = "a+"
        for (tier1Key, tier1Item) in six.iteritems(buffer_feature_matrix):
            tier2BufferDict = {}
            for (tier2Key, tier2Item) in six.iteritems(tier1Item):
                ts_p20, ts_p11, ts_p02 = get_components(tier2Item)
                ts_invar = __invaraintScore__(ts_p20, ts_p11, ts_p02)
                tier1BufferList = []
                if tier2BufferDict.__contains__(ts_invar):
                    tier1BufferList = tier2BufferDict.get(ts_invar)
                tier1BufferList.append(tier2Key)
                tier2BufferDict.update({ts_invar:tier1BufferList})
                
            tier2BufferList = sorted(tier2BufferDict.items(), key=itemgetter(0), reverse=True)
            feature_dict = optimization(tier2BufferList)
            tf.logging.info('%s' %tier1Key +'\t original: %s' %len(tier2BufferDict)+'\t reduced: %s'%len(feature_dict))
            pattern_index = generate_feature_map(feature_dict, pattern_index, 
                                                 tier1Key, append_status, appendType, weight_index)
            
            weight_index += 1.0
        tf.logging.info('\n final pattern index: %s' %pattern_index +'\t weight_index: %s'%weight_index)
        return(pattern_index)
    
    def calculate_entropy(self, appendType, pattern_index):
        
        '''
        Calculate invariance for each n-frame
        '''
        append_index = False
        buffer_feature_matrix = {}
        for (tier1Key, tier1Value) in six.iteritems(self.PosDict):
            index=0
            complete_pattern_list = list(map(lambda currVal:currVal.split(), tier1Value.keys()))
            for current_pattern_list in complete_pattern_list:
                entropy_score = self.calculate_probability(
                    current_pattern_list, complete_pattern_list, tier1Value)
                buffer_feature_matrix = self.generate_feature_matrix(
                    buffer_feature_matrix, tier1Key, current_pattern_list, entropy_score)
                '''
                if append_index:
                    self.writeToFeatureFile(tier1Key, current_pattern_list, entropy_score, "a+")
                else:
                    self.writeToFeatureFile(tier1Key, current_pattern_list, entropy_score, "w+")
                append_index = True
                '''
                index +=1
        
        pattern_index = self.__invariant_optimization__(buffer_feature_matrix, appendType, pattern_index)
        
        return(pattern_index)
    
    
