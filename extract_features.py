# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModel
import numpy
import sys

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
import h5py

# BERT_BASE_CASED='bert-base-cased'
# BERT='bert'
# BERT_BASE_UNCASED='bert-base-uncased'
# BERT_LARGE_CASED='bert-large-cased'
# BERT_LARGE_CASED_WHOLEWORD='bert-large-cased-whole-word-masking'
# ROBERTA_LARGE_CASED='roberta-large'
# ROBERTA_BASE='roberta-base'
# XLNET_LARGE='xlnet-large-cased'
# T5='t5-large'
# XLM='xlm-mlm-tlm-xnli15-1024'
# XLM_R_LARGE='xlm-roberta-large'
# XLM_R_LARGE_MIRROR='xlm-roberta-large-mirror'
# XLM_R_BASE='xlm-roberta-base'
# BERT_MULTI_BASE='bert-base-multilingual-cased'
# BERT_MULTI_BASE_UNCASED='bert-base-multilingual-uncased'
# BERT_MULTI_LARGE='bert-large-multilingual-cased'
# BERT_DE_BASE_CASED='bert-base-german-cased'
# BERT_PUBMED='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# BERT_MIRROR='bert_mirrorwic'


# MODELS=[BERT_MULTI_BASE_UNCASED,ROBERTA_BASE, BERT_MIRROR,XLM_R_LARGE_MIRROR,BERT_PUBMED,BERT_DE_BASE_CASED,BERT_MULTI_BASE,BERT_MULTI_LARGE,BERT,BERT_BASE_CASED,BERT_BASE_UNCASED,ROBERTA_LARGE_CASED,BERT_LARGE_CASED,XLNET_LARGE,T5,XLM,BERT_LARGE_CASED_WHOLEWORD,XLM_R_LARGE,XLM_R_BASE]
# MODELNAME2MODEL={BERT_MULTI_BASE_UNCASED:BertModel, ROBERTA_BASE:RobertaModel, BERT_MIRROR:BertModel, XLM_R_LARGE_MIRROR:XLMRobertaModel,BERT_PUBMED:BertModel,BERT_DE_BASE_CASED:BertModel,BERT_MULTI_LARGE:BertModel,BERT_MULTI_BASE:BertModel,BERT:BertModel,BERT_BASE_CASED:BertModel,BERT_LARGE_CASED_WHOLEWORD:BertModel,BERT_BASE_UNCASED:BertModel,BERT_LARGE_CASED:BertModel,ROBERTA_LARGE_CASED:RobertaModel,XLNET_LARGE:XLNetModel,T5:T5Model,XLM:XLMModel,XLM_R_LARGE:XLMRobertaModel,XLM_R_BASE:XLMRobertaModel}
# MODELNAME2TOKENIZERS={ROBERTA_BASE:RobertaTokenizer,BERT_MULTI_BASE_UNCASED:BertTokenizer, BERT_MIRROR:BertTokenizer, XLM_R_LARGE_MIRROR:XLMRobertaTokenizer, BERT_PUBMED:BertTokenizer,BERT_DE_BASE_CASED:BertTokenizer,BERT_MULTI_LARGE:BertTokenizer,BERT_MULTI_BASE:BertTokenizer,BERT:BertTokenizer,BERT_BASE_CASED:BertTokenizer,BERT_LARGE_CASED_WHOLEWORD:BertTokenizer,BERT_BASE_UNCASED:BertTokenizer,BERT_LARGE_CASED:BertTokenizer, ROBERTA_LARGE_CASED:RobertaTokenizer,XLNET_LARGE:XLNetTokenizer,T5:T5Tokenizer,XLM:XLMTokenizer,XLM_R_LARGE:XLMRobertaTokenizer,XLM_R_BASE:XLMRobertaTokenizer}
EOS_NUM=1


def produce_key(sent):
    sent='\t'.join(sent.split())
    sent = sent.replace('.', '$period$')
    sent = sent.replace('/', '$backslash$')
    return sent


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids,orig_to_tok_maps,orig_tokens):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.orig_to_tok_maps=orig_to_tok_maps
        self.orig_tokens=orig_tokens

def tokenize_map(orig_tokens,tokenizer):
    ### Input
    labels = ["NNP", "NNP", "POS", "NN"]

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []

    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens)+1)
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    return bert_tokens,orig_to_tok_map

def convert_examples_to_features(examples, seq_length, tokenizer,args):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        orig_tokens=example.text_a.split()
        tokens_a,orig_to_tok_map_a = tokenize_map(orig_tokens,tokenizer)
        tokens_b = None
        if example.text_b:
            tokens_b,orig_to_tok_map_b = tokenize_map(example.text_b.split(),tokenizer)
            orig_tokens+=example.text_b.split()
            orig_to_tok_map_a+=orig_to_tok_map_b
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                print ('exceed length:',tokens_a)
                continue #skip when the length exceeds
                # tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        if not args.vocab:
            tokens.append("[SEP]")
            input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            if not args.vocab:
                tokens.append("[SEP]")
                input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        while len(orig_to_tok_map_a) < seq_length:
            orig_to_tok_map_a.append(0)

        assert len(orig_to_tok_map_a) == seq_length
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                orig_to_tok_maps=orig_to_tok_map_a,
                orig_tokens=orig_tokens))
    return features


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



def read_examples(input_file,example_batch):
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip().split('\t')[0]
            if line == '' or len(line.split())>400:
                continue
            examples.append(line)
    start=0
    while start < len(examples):
        yield examples[start:start+example_batch]
        start+=example_batch

def get_orig_seq(input_mask_batch):
    seq=[i for i in input_mask_batch if i!=0]
    return seq


def tokenemb2wemb(average_layer_batch,w2token_batch):
    wembs_sent_batch = []
    for sent_i, sent_embed in enumerate(average_layer_batch):
        sent_embed_out = []
        w2token=w2token_batch[sent_i]
        for start,end in w2token:
            sent_embed_out.append(sum(sent_embed[start:end]) / (end-start))
        wembs_sent_batch.append(numpy.array(sent_embed_out))
    return wembs_sent_batch

def tokenid2wordid(input_ids,tokenizer,examples):
    w2token_batch=[]
    input_ids_filtered=[]
    for i,example in enumerate(examples):
        w2token=[]
        input_id=input_ids[i]
        input_start=0
        for w in example.split():
            w_ids=tokenizer.encode(w,add_special_tokens=False)

            if len(w_ids)==0:
                print (w_ids)
                continue
            
            if input_start+len(w_ids)+EOS_NUM > len(input_id):
                break
            while int(w_ids[0])!=int(input_id[input_start]):
                input_start+=1
                if input_start>=len(input_id):
                    logger.warning ('WARNING: wrong tokenisation {0}'.format(example))
                    w2token=None
                break
            if w2token is None:
                continue
            input_end=input_start+len(w_ids)
            w2token.append((input_start,input_end))
            input_start=input_end
        
        print (example.split())
        print (tokenizer.tokenize(example))
        print (w2token)
        print (input_id)
        w2token_batch.append(w2token)
        input_ids_filtered.append(i)
    return w2token_batch,input_ids_filtered



def examples2embeds(examples,tokenizer,model,device,writer,args):
    inputs=tokenizer.batch_encode_plus(examples,max_length=args.max_seq_length,return_attention_mask=True,add_special_tokens=True,pad_to_max_length='right')
    input_ids=torch.tensor(inputs['input_ids'])
    attention_mask=torch.tensor(inputs['attention_mask']).to(device)
    if args.lg:
        language_id = tokenizer.lang2id[args.lg]
        langs = torch.tensor([[language_id] * input_ids.shape[1]] * len(examples)).to(device)

    # print (examples)
    # print (input_ids)
    input_ids=input_ids.to(device)
    model.eval()
    with torch.no_grad():
        w2token_batch,ids_filtered=tokenid2wordid(input_ids,tokenizer,examples)
        input_ids=input_ids[ids_filtered]
        attention_mask=attention_mask[ids_filtered]
        if args.lg:
            all_encoder_layers, _ = model(input_ids=input_ids, langs=langs, attention_mask=attention_mask)[-2:]
        else:
            all_encoder_layers,_=model(input_ids=input_ids,attention_mask=attention_mask)[-2:]
        layer_start,layer_end=int(args.layers.split('~')[0]),int(args.layers.split('~')[1])
        
        average_layer_batch = sum(all_encoder_layers[layer_start:layer_end]) / (layer_end-layer_start)
        try:
            wembs_sent_batch=tokenemb2wemb(average_layer_batch.cpu().detach().numpy(),w2token_batch)
        except:
            print ('ERROR')
            print (average_layer_batch)
            print (examples)
            return None
        for i,sent in enumerate(examples):
            sent=produce_key(sent)

            payload=numpy.array(wembs_sent_batch[i])
            print (payload.shape)
            try:
                if sent in writer:
                    print ('already exist',sent.encode('utf-8'))
                else:
                    if len(sent.split('\t'))==len(payload):
                        writer.create_dataset(sent, payload.shape, dtype='float32', compression="gzip", compression_opts=9,
                                        data=payload)
                    else:
                        print ('WARNING. Wrong tokenisation')
            except OSError as e:
                print(e, sent)


   
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--model", default=None, type=str, required=True,
                        help=" pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--layers", default='1~13', type=str,help='sum over specific layers')
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gpu', type=int,help='specify the gpu to use')
    parser.add_argument('--lg',type=str,default='',help='language id')
    parser.add_argument('--model_type', type=str,default=None,help='model type')
    
    import os
    args = parser.parse_args()
    if not args.model_type:
        args.model_type=args.model

    if args.output_file:
        writer= h5py.File(args.output_file, 'w')
    else:
        model_pre=args.model_type
        if args.model_type!=args.model:
            model_pre+='_'+os.path.basename(os.path.dirname(args.model))
        
        writer=h5py.File(args.input_file+'.'+model_pre+'.ly-'+str(args.layers)+'.hdf5','w')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:{0}".format(args.gpu) if torch.cuda.is_available() and not args.no_cuda and args.gpu>=0 else "cpu")
        n_gpu=1
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    # layer_indexes = [int(x) for x in args.layers.split(",")]
    # assert args.model_type in MODELS
    if args.model.startswith('xlnet'):
        EOS_NUM=2
    tokenizer = AutoTokenizer.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
    model = AutoModel.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
   
       
    example_counter=0
    for examples in read_examples(args.input_file,args.batch_size):
        example_counter+=1
        print ('processed {0} examples'.format (str(args.batch_size*example_counter)))
        examples2embeds(examples,tokenizer,model,device,writer,args)
    writer.close()

if __name__ == "__main__":
    main()
    
