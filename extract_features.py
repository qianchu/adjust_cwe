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


EOS_NUM=1


def produce_key(sent):
    sent='\t'.join(sent.split())
    sent = sent.replace('.', '$period$')
    sent = sent.replace('/', '$backslash$')
    return sent


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
    
