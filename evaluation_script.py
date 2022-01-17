
import os
import numpy as np
from helpers import *
import h5py
import torch
from collections import OrderedDict, defaultdict
from copy import deepcopy
from scipy.stats import spearmanr
from zipfile import ZipFile

def evaluation(trainer_mim,args):
    print ('==original space==')
    eval_wic(args)
    eval_sim('scws',args)
    eval_context_simlex(args,trainer_mim)

    print ('==after adjustiment==')
    eval_wic(args,trainer_mim)
    eval_sim('scws',args,trainer_mim)
    eval_sim('usim',args,trainer_mim)
    eval_context_simlex(args,trainer_mim)


def produce_test_material(test_data_dir,evaluation,base_embed):
    suffix=''
 
    if evaluation in ['scws']:

        test_data_f=os.path.join(test_data_dir,'./scws/scws.txt{0}.__{1}__.hdf5.npy'.format(suffix,base_embed))


        test_f=os.path.join(test_data_dir,'./scws/scws.txt{0}'.format(suffix[4:]))

  
    elif evaluation =='usim':
        test_data_f=os.path.join(test_data_dir,'./usim/usim_en.txt{0}.__{1}.__hdf5.npy'.format(suffix,base_embed))

        test_f=os.path.join(test_data_dir,'./usim/usim_en.txt{0}'.format(suffix[4:]))
    
        
    else:
        raise NotImplementedError
    test_1,test_2,scores=scws_test(test_data_f, test_f)
    return torch.from_numpy(test_1),torch.from_numpy(test_2),scores
    
  



def scws_test(test_data_f,test_f):
    test_data=np.load(test_data_f,allow_pickle=True)[()]
    test_0=[]
    test_1=[]
    scores=[]
    count=0
    for line in open(test_f):
        line=line.strip()
        text, loc, score=line.strip().split('\t')[:3]
        w=text.split()[int(loc)]
        # print (text,loc,score)
        try:
            embed=test_data[produce_key(text)][int(loc)]
        except KeyError:
            print ('NOT FOUND: {0}'.format(text))
        if count==0:
            test_0.append(embed)
            scores.append(score)
            count += 1
        elif count==1:
            test_1.append(embed)
            # print (score,scores[-1])
            assert score==scores[-1]
            count=0

    return test_0,test_1,scores

def wic_test(test_data_f,test_f):
    test_data=np.load(test_data_f,allow_pickle=True)[()]
    test_0=[]
    test_1=[]
    text_0=[]
    text_1=[]
    loc_0=[]
    loc_1=[]
    scores=[]
    count=0
    for line in open(test_f):
        line=line.strip()
        text, loc, score=line.strip().split('\t')[:3]
        # print (text,loc,score)
        try:
            embed=np.mean(test_data[produce_key(text)],axis=0)
            #embed=test_data[produce_key(text)][int(loc)]
        except KeyError:
            print ('NOT FOUND: {0}'.format(text))
        if count==0:
            test_0.append(embed)
            scores.append(score)
            text_0.append(text)
            loc_0.append(loc)
            count += 1
        elif count==1:
            test_1.append(embed)
            text_1.append(text)
            loc_1.append(loc)

            # print (score,scores[-1])
            assert score==scores[-1]
            count=0

    return (torch.from_numpy(np.vstack(test_0,0)),torch.from_numpy(np.vstack(test_1)),scores,text_0,text_1,loc_0,loc_1)




def produce_context_simlex_data(test_data_dir,base_embed):
    test_f=os.path.join(test_data_dir,'./context_simlex/evaluation_kit_final/data/data_en.tsv.out')
    test_data_f=os.path.join(test_data_dir,'./context_simlex/evaluation_kit_final/data/data_en.tsv.out.__{0}.__hdf5.npy'.format(base_embed))
    return test_f,test_data_f


def produce_wic_data(test_data_dir,base_embed,lg='en'):
    train_data_f = os.path.join(test_data_dir,'./WiC_dataset/train/train.data.txt.out__{0}__.hdf5.npy'.format(base_embed))
    train_f = os.path.join(test_data_dir,  './WiC_dataset/train/train.data.txt.out')
    dev_data_f=os.path.join(test_data_dir,'./WiC_dataset/dev/dev.data.txt.out__{0}__.hdf5.npy'.format(base_embed))
    dev_f=os.path.join(test_data_dir,'./WiC_dataset/dev/dev.data.txt.out')
    test_data_f=os.path.join(test_data_dir,'./WiC_dataset/test/test.data.txt.out__{0}__.hdf5.npy'.format(base_embed))
    test_f=os.path.join(test_data_dir,'./WiC_dataset/test/test.data.txt.out')

    data_scores={}
    data_scores['train'] = wic_test(train_data_f, train_f)
    data_scores['dev'] = wic_test(dev_data_f, dev_f)
    data_scores['train']= wic_test(test_data_f, test_f)

    return data_scores



def eval_sim( testset, args,trainer_mim=None):
    with torch.no_grad():
       
        test_1, test_2, scores = produce_test_material(args.test_data_dir, testset, args.base_embed,
                                        )
     
        mean=args.tgt_data_mean
        normalize_embeddings(test_1,args.norm,mean=mean)
        normalize_embeddings(test_2,args.norm,mean=mean)
        
        if trainer_mim is not None:
            test_1,test_2=trainer_mim.model_tgt(test_1.to(self.device)),trainer_mim.model_tgt(test_2.to(self.device))
        
        scores_pred = produce_cosine_list(test_1, test_2)
        rho = spearmanr(scores, scores_pred)[0]
            
        print('spearman rank for original embedding {0} is {1}'.format(testset, rho))
           
        scores_pred = produce_cosine_list(test_1, test_2)
        rho = spearmanr(scores, scores_pred)[0]
        print(produce_results(testset, 'spearman rank', rho))
    return rho

def thres_search(scores_pred,golds):

    thres=scores_pred[np.argmax(scores_pred)]
    thres_min=scores_pred[np.argmin(scores_pred)]
    num_corrects_prevmax=-1
    num_corrects=0
    thres_max=0
    while thres>=thres_min:
        if num_corrects>num_corrects_prevmax:
            num_corrects_prevmax=num_corrects
            thres_max=thres
        scores_pred_label = np.array(['F'] * len(scores_pred))
        thres-=0.01

        scores_true_indexes = np.where(scores_pred>thres)

        scores_pred_label[scores_true_indexes]='T'
        corrects_true = np.where((np.array(scores_pred_label) == 'T') & (np.array(golds) == 'T'))[0]
        corrects_false=np.where((np.array(scores_pred_label) == 'F') & (np.array(golds) == 'F'))[0]
        num_corrects=len(corrects_true)+len(corrects_false)
        print ('thres: {0}, num of correct: {1}, percentage is" {2}'.format(thres,num_corrects,num_corrects/len(scores_pred)))
    return thres_max


def eval_wic_cosine(scores_pred,golds,thres=None):
    scores_pred,golds=np.array(scores_pred),np.array(golds)
    if thres:
        scores_pred_label = np.array(['F'] * len(scores_pred))
        scores_true_indexes = np.where(scores_pred > thres)

        scores_pred_label[scores_true_indexes] = 'T'
        corrects_true = np.where((np.array(scores_pred_label) == 'T') & (np.array(golds) == 'T'))[0]
        corrects_false = np.where((np.array(scores_pred_label) == 'F') & (np.array(golds) == 'F'))[0]
        num_corrects = len(corrects_true) + len(corrects_false)


        print ('==WIC RESULTS==: thres: {0}, num of correct: {1}, percentage: {2}'.format(thres,num_corrects,num_corrects/len(scores_pred)))

    else:
        thres=thres_search(scores_pred,golds)
        thres,scores_pred_label=eval_wic_cosine(scores_pred,golds,thres)
    return thres,scores_pred_label


def csimlex_test_data(args,base_embed):
    test_f,test_data_f=produce_context_simlex_data(args.test_data_dir,base_embed)

    test_data=np.load(test_data_f,allow_pickle=True)[()]
    test_0=[]
    test_1=[]
    for line in open(test_f):
        line=line.strip()
        text, loc1, loc2=line.strip().split('\t')[:3]
        try:
            embed1=test_data[produce_key(text)][int(loc1)]
            test_0.append(embed1)
            embed2=test_data[produce_key(text)][int(loc2)]
            test_1.append(embed2)
        except KeyError:
            print ('NOT FOUND: {0}'.format(text))

    return torch.from_numpy(np.vstack(test_0,0)),torch.from_numpy(np.vstack(test_1,0))

def output_csimlex(scores_pred,trainer_mim,args):
    subtask1=['change']
    subtask2=['sim_context1\tsim_context2']
    test_dir=os.path.join(args.test_data_dir,'context_simlex')
    model_dir=os.path.join(test_dir,args.base_embed)
    if trainer_mim:
        model_dir+='.adjusted'
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        os.mkdir(os.path.join(model_dir,'res1'))
        os.mkdir(os.path.join(model_dir,'res2'))

    res1dir=os.path.join(model_dir,'res1')
    res2dir=os.path.join(model_dir,'res2')

    with open(os.path.join(res1dir,'results_subtask1_en.tsv'),'w') as f1, open(os.path.join(res2dir,'results_subtask2_en.tsv'),'w') as f2:
        for pair_id in range(int(len(scores_pred)/2)):
            pair_id_start=pair_id*2
            pair_id_end=pair_id*2+1
            score1=scores_pred[pair_id_start]
            score2=scores_pred[pair_id_end]
            change=str(score2-score1)
            subtask1.append(change)
            subtask2.append('{0}\t{1}'.format(str(score1),str(score2)))
        f1.write('\n'.join(subtask1)+'\n')
        f2.write('\n'.join(subtask2)+'\n')



def eval_context_simlex(args,trainer_mim=None):
    with torch.no_grad():
        data1,data2=csimlex_test_data(args,args.base_embed)

        mean=args.tgt_data_mean
        normalize_embeddings(data1,args.norm,mean=mean)
        normalize_embeddings(data2,args.norm,mean=mean)
        
        if trainer_mim is not None:
            data1=trainer_mim.model_tgt(data1.to(self.device))
            data2=trainer_mim.model_tgt(data2.to(self.device))

            
        scores_pred = produce_cosine_list(data1, data2)
        output_csimlex(scores_pred,trainer_mim,args)

def output_wic_test(test_pred,args,trainer_mim):
    wic_test_dir=os.path.join(args.test_data_dir,'./WiC_dataset/test/'+args.base_embed)
    if trainer_mim is not None:
        wic_test_dir+'.adjusted'
    if not os.path.exists(wic_test_dir):
        os.mkdir(wic_test_dir)
    
    with open(os.path.join(wic_test_dir, 'output.txt'), 'w') as f:
        f.write('\n'.join(test_pred))


def eval_wic(args,trainer_mim=None):
    with torch.no_grad():
        
        data_scores=produce_wic_data(args.test_data_dir,args.base_embed)
        wicdata={}
        wicdata['train_1'],wicdata['train_2'],train_scores,_,_,_,_=data_scores['train']
        wicdata['dev_1'],wicdata['dev_2'],dev_scores,_,_,_,_=data_scores['dev']
        wicdata['test_1'],wicdata['test_2'],test_scores,_,_,_,_=data_scores['test']
        print ('apply model ')

        mean=args.tgt_data_mean
        for d in wicdata:
            normalize_embeddings(wicdata[d],args.norm,mean=mean)
       
        for d in wicdata:
            if trainer_mim is not None:
                wicdata[d]=trainer_mim.model_tgt(wicdata[d].to(self.device))
            
 
        train_scores_pred = produce_cosine_list(d['train_1'], d['train_2'])
        print ('==WIC RESULTS==: average cosine for training',sum(train_scores_pred)/len(train_scores_pred),len(train_scores_pred))
        dev_scores_pred = produce_cosine_list(d['dev_1'], d['dev_2'])

        print ('train_dev results for wic ')
       
        train_dev_scores_pred=train_scores_pred+dev_scores_pred
        train_dev_scores=train_scores+dev_scores
      
        train_dev_thres, train_dev_pred = eval_wic_cosine(train_dev_scores_pred, train_dev_scores)

        print ('test results for wic with traindev thres')

        scores_pred = produce_cosine_list(d['test_1'], d['test_2'])
        thres, test_pred=eval_wic_cosine(scores_pred,test_scores,train_dev_thres)

        print ('train results for wic with traindev thres')
        _,_=eval_wic_cosine(train_scores_pred,train_scores,train_dev_thres)

        print ('dev results for wic with traindev thres')
        _,_=eval_wic_cosine(dev_scores_pred,dev_scores,train_dev_thres)

        output_wic_test(test_pred,args,trainer_mim)








