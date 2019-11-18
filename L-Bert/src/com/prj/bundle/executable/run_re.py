# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""L-BERT implementation - sentence level classification task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os, sys
import tensorflow as tf
import re
import time
import numpy as np
import socket
import re
from operator import itemgetter
import six
from sklearn import svm

if socket.gethostname() == 'GPU_1':
    sys.path.append('../L-Bert')
elif  socket.gethostname() == 'GPU_2':
    sys.path.append('../L-Bert')


from src.com.prj.bundle.modelling import modeling
from src.com.prj.bundle.optimization import optimization, featureInvariance
from src.com.prj.bundle.tokenization import tokenization


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("pretrain_embed", None,
                    "Pre-trained word embedding")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "feature_dir", None,
    "Invariant semantic features learned on training corpus.")

flags.DEFINE_list(
    'layer_def', None,
    "Layer setup for the input embedding")

flags.DEFINE_integer(
    'test_fold', None,
    "test prediction repetition number")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("kernel_size", 3,
                   "Window size to convolve over sequence")

flags.DEFINE_integer("stride_size", 1,
                   "Slide size for window")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

      When running eval/predict on the TPU, we need to pad the number of examples
      to be a multiple of the batch size, because the TPU requires a fixed batch
      size. The alternative is to drop the last batch, which is bad because it means
      the entire output data won't be generated.
    
      We use this class instead of `None` because treating `None` as padding
      batches could cause silent errors.
      """


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self,
                   input_ids,
                   cluster_ids,
                   context_ids,
                   input_mask,
                   context_mask,
                   segment_ids,
                   label_id,
                   is_real_example=True):
        self.input_ids = input_ids
        self.cluster_ids = cluster_ids
        self.context_ids = context_ids
        self.input_mask = input_mask
        self.context_mask = context_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class LBERTProcessor(DataProcessor):
            
    def get_train_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "train"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "train")
        
    def get_dev_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "dev"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "dev")
        
    def get_test_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "test"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "test")
        
    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                        
        return examples
    
    
class CLBERTProcessor(DataProcessor):
            
    def get_train_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "train"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "train")
        
    def get_dev_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "dev"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "dev")
        
    def get_test_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "test"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "test")
        
    def get_labels(self):
        """See base class."""
        return ["cpr:3", "cpr:4", "cpr:5", "cpr:6", "cpr:9", "false"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "false"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                       
        return examples
    
class BioBERTGeniaProcessor(DataProcessor):
    def get_train_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "train"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "train")
        
    def get_dev_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "dev"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "dev")
        
    def get_test_examples(self, data_dir, text_type):
        """See base class."""
        localeFile = "test"+text_type+".tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, localeFile)), "test")
        
    def get_labels(self):
        """See base class."""
        return ["1", "2","false"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "false"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            
        '''
        for entry in examples:
            print(entry.guid,'\t',entry.label,'\n',entry.text_a,'\n')
        '''
            
        return examples

    
def uniform_padding(input_bool, padding, buffer_list, max_seq_length):
    
    if input_bool:
        while len(buffer_list) < max_seq_length:
            buffer_list.append(padding)
        assert len(buffer_list) == max_seq_length
    else:
        buffer_list = list(map(lambda currVal : -1 if currVal >= 0 else -1, list(range(0,max_seq_length))))
    
    return(buffer_list)
        

def feature_doprout(seq_activation, ex_index):
    
    dropout_rate = 0.01
    seq_dict = dict(enumerate(seq_activation))
    seq_set = sorted(set(seq_activation), reverse=True)
    seq_length = len(seq_set)
    drop_index = seq_length-int(np.round(seq_length*dropout_rate))
    drop_values = seq_set[drop_index:seq_length]
    for (key,value) in six.iteritems(seq_dict):
        if value in drop_values:
            seq_dict.update({key:-1.0})
    
    update_dict = dict(filter(lambda currVal : currVal[1] != -1.0, seq_dict.items()))
    relevant_indices = list(update_dict.keys())
    seq_activation = list(update_dict.values())
    if len(update_dict) < len(seq_dict):
        a=1
    else:
        a=1
       
    return(relevant_indices, seq_activation)

def pop_feature(index_list, decoy_list):
    
    return_list = []
    for index,item in enumerate(decoy_list):
        if index in index_list:
            return_list.append(item)
    
    return(return_list)


def convert_single_example(ex_index, example, pos_example, chunk_example, 
                           label_list, max_seq_length, instance, tokenizer, 
                           con_tokenizer, layer_def):
    
    """Converts a single `InputExample` into a single `InputFeatures`."""
    
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            cluster_ids=[0] * max_seq_length,
            context_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            context_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    '''
    Generate n-frame context based features from the raw text
    '''
    posCluster_map_a, chunkCluster_map_a, context_a, seq_activation_a = con_tokenizer.convert_to_context(
        example.text_a, pos_example.text_a, chunk_example.text_a, max_seq_length, instance)

    context_b = None    
    if example.text_b:
        posCluster_map_b, chunkCluster_map_b, context_b, seq_activation_b = con_tokenizer.convert_to_context(
        example.text_b, pos_example.text_b, chunk_example.text_b, max_seq_length, instance)
        

    temp_list = []
    for item in list(example.text_a.split()):
        if re.match('^BENTITY1', item):
            temp_list.append(item)
        elif re.match('^BENTITY2', item):
            temp_list.append(item)
        else:
            temp_list.append(item)
            
    text_a = ' '.join(i for i in temp_list)
    tokens_a, posTokens_a = tokenizer.tokenize(text_a)
    if len(tokens_a) != len(posTokens_a):
        tf.logging.info('Inconsistent token size %s' %len(tokens_a) +' and %s ' %len(posTokens_a))
    tokens_b = None
    if example.text_b:
        temp_list = []
        for item in list(example.text_b.split()):
            if re.findall('BENTITY\d+', item):
                temp_list.append(item)
            else:
                temp_list.append(item)
        text_b = ' '.join(i for i in temp_list)
        tokens_b, posTokens_b = tokenizer.tokenize(text_b)

    if context_b:
        _truncate_seq_pair(context_a, context_b, max_seq_length - 3)
    else:
        if len(context_a) > max_seq_length - 2:
            context_a = context_a[0:(max_seq_length - 2)]
        
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
            posTokens_a = posTokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
    
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
        tokens.append("[SEP]")
    '''
    Appending tokenized POS contexts for original lexical feature representation
    '''
    contexts = []
    contexts.append("[CLS]")
    for context in context_a:
        contexts.append("["+(','.join(value for value in context))+"]")
    contexts.append("[SEP]")
    
    if context_b:
        for context in context_b:
            contexts.append("["+(','.join(value for value in context))+"]")
        contexts.append("[SEP]")
    
    '''
    Secondary activation over layer over lexical features (Optional)
    '''
    activations = []
    activations.append(1.0)
    for mapped_value in posTokens_a:
        activations.append(1.0)
    activations.append(1.0)
    
    if context_b:
        for mapped_value in posTokens_b:
            activations.append(1.0)
        activations.append(1.0)

    input_ids = []
    if bool(layer_def[0]):
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

    '''
    Lexical Embedding (Pos tag cluster feature based) : Appending POS lexical context id to map features
    '''
    cluster_ids = []
    if bool(layer_def[1]):
        cluster_ids.append(7)
        for mapped_value in posTokens_a:
            cluster_ids.append(posCluster_map_a[mapped_value])
        cluster_ids.append(8)
        
        if context_b:
            for mapped_value in posTokens_b:
                cluster_ids.append(posCluster_map_b[mapped_value])
            cluster_ids.append(8)
       
    '''
    (Optional)
    Lexical Embedding (Chunk tag Cluster feature based) : Appending Chunk Tag lexical context id to map features
    '''
    context_ids = []
    if bool(layer_def[1]):
        context_ids.append(7)
        for mapped_value in posTokens_a:
            context_ids.append(chunkCluster_map_a[mapped_value])
        context_ids.append(8)
        
        if context_b:
            for mapped_value in posTokens_b:
                context_ids.append(chunkCluster_map_b[mapped_value])
            context_ids.append(8)
            
    # segment id's
    segment_ids = []
    if bool(layer_def[2]) and bool(layer_def[0]):
        cls_tag = True
        for value in tokens:
            if value == '[CLS]' and cls_tag:
                cls_tag = False
                append_value=0
            elif value == '[CLS]' and not(cls_tag):
                cls_tag = True
                append_value=1
            segment_ids.append(append_value)
    elif bool(layer_def[2]) and bool(layer_def[1]):
        cls_tag = True
        for value in contexts:
            if value == '[CLS]' and cls_tag:
                cls_tag = False
                append_value=0
            elif value == '[CLS]' and not(cls_tag):
                cls_tag = True
                append_value=1
            segment_ids.append(append_value)
  
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = []
    if bool(layer_def[3]) and bool(layer_def[0]):
        input_mask = [1.0] * len(tokens)
        
    '''
    (Optional)
    Activation mask for lexical features
    '''
    context_mask = []
    if bool(layer_def[1]):
        context_mask.append(1)
        for mapped_value in posTokens_a:
            if(context_a[mapped_value][0] == "BENTITY1" 
               or context_a[mapped_value][0] == "BENTITY2"): 
                context_mask.append(2)
            else:
                context_mask.append(1)
        context_mask.append(1)
        
        if context_b:
            for mapped_value in posTokens_b:
                if(context_b[mapped_value][0] == "BENTITY1" 
                   or context_b[mapped_value][0] == "BENTITY2"):
                    context_mask.append(0)
                else:
                    context_mask.append(1)
            context_mask.append(1)
        

    if len(input_ids) != len(cluster_ids) and len(input_ids) != len(context_ids):
        tf.logging.info('%s' %len(input_ids) +'\t %s' %len(cluster_ids) +'\t %s' %len(context_ids))
        tf.logging.info('%s' %len(tokens)+'\t %s' %len(posTokens_a))
        tf.logging.info('%s' %tokens +'\n %s' %input_ids +'\n %s' %cluster_ids +'\n %s' %context_ids)
    
    
    # Zero-pad up to the sequence length.
    input_ids = uniform_padding(bool(layer_def[0]), 0, input_ids, max_seq_length)
    cluster_ids = uniform_padding(bool(layer_def[1]), 0, cluster_ids, max_seq_length)
    context_ids = uniform_padding(bool(layer_def[1]), 0, context_ids, max_seq_length)
    segment_ids = uniform_padding(bool(layer_def[2]), 0, segment_ids, max_seq_length)
    input_mask = uniform_padding(bool(layer_def[3]), 0.0, input_mask, max_seq_length)
    context_mask = uniform_padding(bool(layer_def[3]), 0, context_mask, max_seq_length)

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([
            tokenization.printable_text(tokenization.manage_print(x)) for x in tokens]))
        tf.logging.info("activations: %s" % " ".join([str(x) for x in activations]))
        tf.logging.info("contexts: %s" % " ".join([
            tokenization.printable_text(tokenization.manage_print(x)) for x in contexts]))
        
        bool(layer_def[0]) and tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        bool(layer_def[1]) and  tf.logging.info("cluster_ids: %s" % " ".join([str(x) for x in cluster_ids]))
        bool(layer_def[1]) and  tf.logging.info("context_ids: %s" % " ".join([str(x) for x in context_ids]))
        bool(layer_def[2]) and  tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        bool(layer_def[3]) and  tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        bool(layer_def[3]) and  tf.logging.info("context_mask: %s" % " ".join([str(x) for x in context_mask]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
          input_ids=input_ids,
          cluster_ids=cluster_ids,
          context_ids=context_ids, 
          input_mask=input_mask,
          context_mask = context_mask,
          segment_ids=segment_ids,
          label_id=label_id,
          is_real_example=True)
    
    return feature


def file_based_convert_examples_to_features(
    examples, pos_examples, chunk_examples, label_list, max_seq_length, embed_locale, 
    feature_locale, layer_def, tokenizer, con_tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    
    instance = featureInvariance.SemanticInvariance(embed_locale, feature_locale)
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        pos_example = pos_examples[ex_index]
        chunk_example = chunk_examples[ex_index]
        feature = convert_single_example(ex_index, example, pos_example, chunk_example, 
                                         label_list, max_seq_length, instance, tokenizer, 
                                         con_tokenizer, layer_def)

        def create_int_feature(values):

            if any(list(map(lambda currVal : isinstance(currVal, float), list(values)))):
                f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            else:
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["cluster_ids"] = create_int_feature(feature.cluster_ids)
        features["context_ids"] = create_int_feature(feature.context_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["context_mask"] = create_int_feature(feature.context_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
    
        # creates data_set in a Json format
        # a json file for each example
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
                
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
  
    ''' generates an input shape information using dimension and data type'''
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "cluster_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "context_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "context_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        }
    
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            if t.dtype == float:
                t = tf.to_float(t)
            example[name] = t

        return example
    
    def input_fn(params):
        """The actual input function."""
        tf.logging.info("*********loading input from file_based_input_fn_builder()*********")
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def assert_rank(tensor, expected_rank, name=None):
    
    """Raises an exception if the tensor rank is not of the expected rank.

      Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.
    
      Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_shape_list(tensor, expected_rank=None, name=None):
    
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
          specified and the `tensor` has a different rank, and exception will be
          thrown.
        name: Optional name of the tensor for the error message.
    
      Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
      """
      
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def create_model(bert_config, is_training, input_ids, cluster_ids, context_ids, 
                 input_mask, context_mask, segment_ids, labels, num_labels, 
                 use_one_hot_embeddings, layer_def, feature_locale, kernel_size):
    
    """Creates a classification model."""
    model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          layer_def=layer_def,
          feature_locale=feature_locale,
          kernel_size=kernel_size,
          input_ids=input_ids,
          cluster_ids=cluster_ids,
          context_ids=context_ids,
          input_mask=input_mask,
          context_mask = context_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    #output_layer = model.get_sequence_output()
    
    if bool(layer_def[1]) and bool(layer_def[0]):
        #output_layer = model.get_concat_output()
        #output_layer = model.get_pooled_output()
        output_layer = model.get_sequence_output()
        #cluster_layer = model.get_cluster_output()
        #word_output = model.get_word_output()
    elif bool(layer_def[1]):
        output_layer = model.get_reduced_output()
    elif bool(layer_def[0]):
        output_layer = model.get_pooled_output()
    
    input_shape = get_shape_list(output_layer, expected_rank=3)

    '''
    Logsum evaluation of the terminal outputs from [CLS] and [SEP] tokens
    '''
    first_token_tensor = tf.squeeze(output_layer[:, 0:1, :], axis=1)
    first_token_tensor = tf.reshape(first_token_tensor, [input_shape[0], 1, input_shape[2]])
    index = tf.where(tf.equal(input_ids, 102))
    second_token_tensor = tf.gather_nd(output_layer, index)
    second_token_tensor = tf.reshape(second_token_tensor, [input_shape[0], 1, input_shape[2]])
    output_layer = tf.concat([first_token_tensor,second_token_tensor],1)
    output_layer = tf.reduce_logsumexp(output_layer, reduction_indices=[1])
    output_layer = tf.reshape(output_layer, [input_shape[0], input_shape[2]])

    #output_layer = tf.norm(output_layer, axis=1)
    #context_mask = tf.cast(context_mask, tf.float32)
    #output_layer = tf.concat([output_layer, context_mask], 1)
    
    #hidden_size = logits.shape[-1].value
    hidden_size = output_layer.shape[-1].value
    
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())
    
    #tf.logging.info('op: %s' %output_layer +'\n weight: %s' %output_weights +'\n bias: %s' %output_bias)
  
    with tf.variable_scope("loss"):

        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.95)
    
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        
    
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    
    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, layer_def, kernel_size, 
                     feature_locale, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    
    """Returns `model_fn` closure for TPUEstimator."""
    
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            
        #tf.logging.info('FEATURE %s' %features["label_ids"])
        input_ids = features["input_ids"]
        cluster_ids = features["cluster_ids"]
        context_ids = features["context_ids"]
        input_mask = features["input_mask"]
        context_mask = features["context_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
            
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, cluster_ids, context_ids, input_mask, 
            context_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings, 
            layer_def, feature_locale, kernel_size)

        tf.logging.info(%total_loss)
        
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)
      
        #return logits
    
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
                        
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
      
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    
    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })
        
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        
        return d
    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            
        feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    
    start_time = time.time()
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    processors = {
        "isp":LBERTProcessor,
        "cisp":CLBERTProcessor,
        "pisp": BioBERTGeniaProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                    FLAGS.init_checkpoint)
    
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    
    tf.gfile.MakeDirs(FLAGS.output_dir)
    
    task_name = FLAGS.task_name.lower()
    
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    
    processor = processors[task_name]()
    
    label_list = processor.get_labels()
    
    layer_def = list(map(lambda currVal : int(currVal), FLAGS.layer_def))
    
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    con_tokenizer = tokenization.ContextTokenizer(
        kernel_size=FLAGS.kernel_size, stride_size=FLAGS.stride_size)
    
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    
    # Gather Training Examples
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir, '')
        trainPos_examples = processor.get_train_examples(FLAGS.data_dir, 'Pos')
        trainChunk_examples = processor.get_train_examples(FLAGS.data_dir, 'Chunk')
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        #num_train_steps = 100000
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        layer_def=layer_def,
        kernel_size=FLAGS.kernel_size,
        feature_locale=FLAGS.feature_dir,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, trainPos_examples, trainChunk_examples, label_list, 
            FLAGS.max_seq_length, FLAGS.pretrain_embed, FLAGS.feature_dir, 
            layer_def, tokenizer, con_tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    
    if FLAGS.do_eval:
        print('CALLED EVAL now')
        eval_examples = processor.get_dev_examples(FLAGS.data_dir, '')
        evalPos_examples = processor.get_dev_examples(FLAGS.data_dir, 'Pos')
        evalChunk_examples = processor.get_dev_examples(FLAGS.data_dir, 'Chunk')
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, evalPos_examples, evalChunk_examples, label_list, 
            FLAGS.max_seq_length, FLAGS.pretrain_embed, FLAGS.feature_dir, 
            layer_def, tokenizer, con_tokenizer, eval_file)
    
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)
    
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
    
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    
        
    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir, '')
        predictPos_examples = processor.get_test_examples(FLAGS.data_dir, 'Pos')
        predictChunk_examples = processor.get_test_examples(FLAGS.data_dir, 'Chunk')
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())
    
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples, predictPos_examples, predictChunk_examples, label_list, 
            FLAGS.max_seq_length, FLAGS.pretrain_embed, FLAGS.feature_dir, 
            layer_def, tokenizer, con_tokenizer, predict_file)
    
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
    
        result = estimator.predict(input_fn=predict_input_fn)
    
    
        output_predict_file = os.path.join(FLAGS.data_dir, "test_results"+str(FLAGS.test_fold)+".tsv")
        with tf.gfile.GFile(output_predict_file, "w+") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

    end_time = (time.time()-start_time)*100
    tf.logging.info('total execution time %f'%end_time+" seconds")

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
