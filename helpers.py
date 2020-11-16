from variable_names import *
import torch
import numpy as np
import os
import h5py
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import spacy

nlp_es = spacy.blank('es')

def lemmatize(w,lg,en_pos=None):
    if lg=='en':
        if not en_pos:
            w = lemmatizer.lemmatize(w)
        else:
            w=lemmatizer.lemmatize(w,en_pos)
    if lg=='es':
        # for token in nlp(w):
        w=nlp_es(w)[0].lemma_
    return w

def permutate(w_lst):
    l=[]
    for i,w in enumerate(w_lst):
        rem_lst=w_lst[i+1:]
        for rem_w in rem_lst:
            l.append([w,rem_w])
    return l

def produce_results(testset,explanation,result):
    return 'TESTRESULT:{0}:{1}:::{2}'.format(testset,explanation,result)

def extract_emb(zh_data_f_lst,zh_line):
    zh_line_data=None
    # start = timer()


    zh_line_key=produce_key(zh_line)
    # end = timer()
    # print ('produce key',end-start)
    for zh_data_f in zh_data_f_lst:
        # start=timer()
        try:
            zh_line_data = zh_data_f[zh_line_key][:]
        except KeyError as e:
           print ('extraction from training:',e)
        # end=timer()
        # print ('extract embed from h5py',end-start)
    if type(zh_line_data)==type(None):
        print('Not in data: {0}'.format(zh_line))
    return zh_line_data

def produce_cosine_list(test_src,test_tgt):
    cos_matrix=produce_cos_matrix(test_src, test_tgt)
    scores_pred = [float(cos_matrix[i][i]) for i in range(len(cos_matrix))]
    return scores_pred

def produce_cos_matrix(test_src,test_tgt):
    normalize_embeddings(test_src, NORMALIZE, None)
    normalize_embeddings(test_tgt, NORMALIZE, None)
    cos_matrix = torch.mm(test_src, test_tgt.transpose(0, 1))
    return cos_matrix




def convert_h5py_2dict(h5pyfile):
    sent2emb={}
    sent2index=yaml.safe_load(h5pyfile['sent2index'][0])
    data=h5pyfile['data'][:]
    for key in sent2index:
        start,end=sent2index[key]
        try:
            sent2emb[key]=data[start:end]
        except KeyError as e:
            print (e)
    return sent2emb



def h5pyfile_lst(zh_data_lst):
    zh_data_f_lst=[]
    for zh_data in zh_data_lst:
        zh_data_f=np.load(zh_data,allow_pickle=True)[()]
        if 'sent2index' in zh_data_f:
            zh_data_f=convert_h5py_2dict(zh_data_f)
        zh_data_f_lst.append(zh_data_f)
    return zh_data_f_lst

def close_h5pyfile_lst(zh_data_lst):
    for f in zh_data_lst:
        if type(f)==h5py.File:
            f.close()
def matrix_norm(emb):
    emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))

def produce_key(sent):
    sent='\t'.join(sent.split())
    sent = sent.replace('.', '$period$')
    sent = sent.replace('/', '$backslash$')
    return sent


def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == CENTER:
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == NORMALIZE:
            matrix_norm(emb)
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean if mean is not None else None

def retrieve_data_with_loc(test_data,key_text,loc):
    data=test_data[key_text]
    if loc=='all':
        if len(data)>1:
            data=sum(data)/len(data)
        else:
            data=data[0]
    else:
        data=test_data[key_text][loc]
    return data


def construct_dict(test_data,w_list=None):
    emb_list=[]
    word2index={}
    index2word={}
    counter=0

    if w_list:
        keys=w_list
    else:
        keys=sorted(test_data.keys())
    for key in keys:
        key_text=key
        loc = 'all'
        if type(key)==tuple:
            key_text=key[0]
            loc=int(key[1])
    #     if key in test_data:
    # for key in test_data:
        try:
            # emb_list.append(test_data[key_text][loc])
            emb_list.append(retrieve_data_with_loc(test_data,key_text,loc))
        except KeyError as e:
            continue
        word2index[key]=counter
        index2word[counter]=key
        # print (key)
        counter+=1
            # if counter>2000:
            #     break
        # else:
        #     print (key,'not found')
        if counter>=10000 and counter%10000==0:
            print ('constructing dictionary line',counter)
    print ('candidates no {0}'.format(len(emb_list)))
    return emb_list,word2index,index2word
