
import os
import numpy as np
from variable_names import *
from helpers import *
import h5py
import torch
from collections import OrderedDict, defaultdict
from copy import deepcopy
from scipy.stats import spearmanr
from zipfile import ZipFile


def produce_test_material(test_data_dir,evaluation,base_embed,avg_flag,lg,context_num=''):
    suffix=''
    if evaluation.endswith('.vocab'):
        suffix = '.txt.vocab'

    if evaluation in ['scws','scws.vocab']:

        test_data_f=os.path.join(test_data_dir,'./scws/scws.txt{0}.{1}.hdf5.npy'.format(suffix,MODEL2FNAME[base_embed]['en']))

        if evaluation =='scws.vocab' and avg_flag:
            test_data_f = os.path.join(test_data_dir, './scws/average_scws_en.txt.vocab__anchor_en.txt.parallel.{0}.hdf5'.format(MODEL2FNAME[
                base_embed]['en']))

        test_f=os.path.join(test_data_dir,'./scws/scws.txt{0}'.format(suffix[4:]))
        test_1,test_2,scores=scws_test(test_data_f, test_f)

        return test_1,test_2,scores
  


    elif evaluation =='usim':
        test_data_f=os.path.join(test_data_dir,'./usim/usim_en.txt{0}.{1}.hdf5.npy'.format(suffix,MODEL2FNAME[base_embed]['en']))

        test_f=os.path.join(test_data_dir,'./usim/usim_en.txt{0}'.format(suffix[4:]))
        test_1,test_2,scores=scws_test(test_data_f, test_f)
        
        
        return test_1,test_2,scores
    
  



    elif evaluation=='simlex':
        if avg_flag:
            test_data_f=os.path.join(test_data_dir,'./vocab/average_{2}SimLex-999_preprocessed.txt__anchor_{1}.{0}.hdf5'.format(MODEL2FNAME[base_embed]['en'],lg2corpus[lg]['en'],context_num))
        else:
            test_data_f=os.path.join(test_data_dir,'./vocab/SimLex-999_preprocessed.txt.{0}.hdf5'.format(MODEL2FNAME[base_embed]['en']))
        test_f = os.path.join(test_data_dir, './vocab/SimLex-999_preprocessed.txt')
        test_1,test_2,scores=scws_test(test_data_f, test_f)
        return np.vstack(test_1),np.vstack(test_2),scores



def file2generator(fname):
    for line in open(fname):
        yield line


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

    return test_0,test_1,scores,text_0,text_1,loc_0,loc_1

def lexsub_check(lex2subs,lex,word):
    if not lex2subs:
        return True
    else:
        if word in lex2subs.get(lex):
            return True
        else:
            return False







def test_f2test_list(test_f,testset):
    zh_w_list=[]
    en_w_list=[]
    for line in open(test_f):

        zh_w, en_w = line.strip().split(' ||| ')
        if testset in [NP_CLS,NP_CLS_type,NP_CLS_WSD, NP_CLS_WSD_type]:
            zh_w=zh_w.split('|&|')[0].split('::')[0]
        if testset in [NP_CLS_type,NP_CLS_WSD_type]:
            en_w=en_w.split('|&|')[0].split('::')[0]
        zh_w_list.append(produce_key(zh_w))
        en_w_list.append(produce_key(en_w))

    en_w_list=list(OrderedDict.fromkeys(en_w_list).keys())
    zh_w_list=list(OrderedDict.fromkeys(zh_w_list).keys())
    return zh_w_list,en_w_list



def produce_context_simlex_data(test_data_dir,base_embed):
    test_f=os.path.join(test_data_dir,'./context_simlex/evaluation_kit_final/data/data_en.tsv.out')
    test_data_f=os.path.join(test_data_dir,'./context_simlex/evaluation_kit_final/data/data_en.tsv.out.{0}.hdf5.npy'.format(MODEL2FNAME[base_embed]['en']))
    return test_f,test_data_f


def produce_wic_data(test_data_dir,base_embed,lg='en'):
    train_data_f = os.path.join(test_data_dir,'./WiC_dataset/train/train.data.txt.out.{0}.hdf5.npy'.format(MODEL2FNAME[base_embed][lg]))
    train_f = os.path.join(test_data_dir,  './WiC_dataset/train/train.data.txt.out')
    dev_data_f=os.path.join(test_data_dir,'./WiC_dataset/dev/dev.data.txt.out.{0}.hdf5.npy'.format(MODEL2FNAME[base_embed][lg]))
    dev_f=os.path.join(test_data_dir,'./WiC_dataset/dev/dev.data.txt.out')
    test_data_f=os.path.join(test_data_dir,'./WiC_dataset/test/test.data.txt.out.{0}.hdf5.npy'.format(MODEL2FNAME[base_embed][lg]))
    test_f=os.path.join(test_data_dir,'./WiC_dataset/test/test.data.txt.out')

    train_1, train_2, train_scores,train_1_text,train_2_text,train_1_text_loc,train_2_text_loc = wic_test(train_data_f, train_f)
    dev_1, dev_2, dev_scores,dev_1_text,dev_2_text,dev_1_text_loc,dev_2_text_loc = wic_test(dev_data_f, dev_f)
    test_1, test_2, test_scores, test_1_text, test_2_text, test_1_text_loc,test_2_text_loc= wic_test(test_data_f, test_f)
    data_scores={'train':(np.vstack(train_1),np.vstack(train_2),train_scores,train_1_text,train_2_text,train_1_text_loc,train_2_text_loc),'dev':(np.vstack(dev_1),np.vstack(dev_2),dev_scores,dev_1_text,dev_2_text,dev_1_text_loc,dev_2_text_loc),'test':(np.vstack(test_1),np.vstack(test_2),test_scores,test_1_text,test_2_text,test_1_text_loc,test_2_text_loc)}

    return data_scores


def neighbour_crossling_sim(src_w, w_embed,matrix_crossling,words_crossling,word_crossling_target,precision,topn=10,error_output_f=None):
    similarity=torch.mm(matrix_crossling,w_embed.reshape(w_embed.size()[0],1)).reshape(matrix_crossling.size()[0])#.detach().cpu().numpy()
    # print (similarity)
    # similarity=matrix_crossling.dot(w_embed)
    # print (word_crossling_target)
    count=1
    found_flag=False
    # print ('sort sim')
    sorted_sim=torch.argsort(-similarity,dim=0)
    # sorted_sim=(-similarity).argsort()
    # print ('iterate')
    for rank,i in enumerate(sorted_sim):
    # for i in (-similarity).argsort():
    #     print('{0}: {1}'.format(str(words_crossling[int(i)]), str(similarity[int(i)])))
        if rank==0:
            top_w=str(words_crossling[int(i)])
            top_cos=similarity[int(i)]
        if str(words_crossling[int(i)]).strip()==word_crossling_target.strip():
            found_flag=True
            sim=float(similarity[int(i)])/float(top_cos)
            if count==1:
                # print ('first spotted')
                precision[1]+=1
                precision[5]+=1
                precision[10]+=1
            elif count<=5:
                precision[5]+=1
                precision[10]+=1
            elif count<=10:
                precision[10]+=1
            break
        count += 1


        # if count == topn:
        #     break

    if error_output_f:
        error_output_f.write('{0}~{1}~{2}~{3}\n'.format(src_w.strip().replace('\n','').replace('\r',''), top_w.strip().replace('\n','').replace('\r',''), word_crossling_target.strip().replace('\n','').replace('\r',''),found_flag))

    return count,sim


def neighbour_crossling(src_w, w_embed,matrix_crossling,words_crossling,word_crossling_target,precision,topn=10,error_output_f=None):
    # similarity=matrix_crossling.dot(np.linalg.norm(w_embed))
    # print (w_embed.size(),matrix_crossling.size())
    # print ('matrix similarity')
    similarity=torch.mm(matrix_crossling,w_embed.reshape(w_embed.size()[0],1)).reshape(matrix_crossling.size()[0])#.detach().cpu().numpy()
    # print (similarity)
    # similarity=matrix_crossling.dot(w_embed)
    # print (word_crossling_target)
    count=1
    sim=0
    found_flag=False
    # print ('sort sim')
    sorted_sim=torch.argsort(-similarity,dim=0)
    # sorted_sim=(-similarity).argsort()
    # print ('iterate')
    for rank,i in enumerate(sorted_sim):
    # for i in (-similarity).argsort():
    #     print('{0}: {1}'.format(str(words_crossling[int(i)]), str(similarity[int(i)])))
        if rank==0:
            top_w=str(words_crossling[int(i)])
            top_cos=similarity[int(i)]
        if str(words_crossling[int(i)]).strip()==word_crossling_target.strip():
            found_flag=True
            sim=float(similarity[int(i)])/float(top_cos)
            if count==1:
                # print ('first spotted')
                precision[1]+=1
                precision[5]+=1
                precision[10]+=1
            elif count<=5:
                precision[5]+=1
                precision[10]+=1
            elif count<=10:
                precision[10]+=1
            break
        count += 1


        if count == topn:
            break

    if error_output_f:
        error_output_f.write('{0}~{1}~{2}~{3}\n'.format(src_w.strip().replace('\n','').replace('\r',''), top_w.strip().replace('\n','').replace('\r',''), word_crossling_target.strip().replace('\n','').replace('\r',''),found_flag))

    return count,sim


def add_baseline(test_src,test_src_other,baseline):
    if baseline=='cat':
        test_src=torch.cat([test_src,test_src_other],dim=1)
    elif base_embed=='avg':
        test_src=(test_src+test_src_other)/2
    return test_src

def eval( testset, args,trainer):
    with torch.no_grad():
        # if testset in EVAL_DATA:
        #     test_1, test_2, scores = deepcopy(EVAL_DATA[testset])
        # else:
        test_1, test_2, scores = produce_test_material(args.test_data_dir, testset, args.base_embed,
                                                        args.avg_flag, args.lg, args.context_num)
            # EVAL_DATA[testset] = deepcopy((test_1, test_2, scores))

        test_src = test_1
        test_tgt = test_2
          
        trainer.model_src.eval()
        if testset.startswith('scws') or testset == 'simlex' or testset.startswith('usim'):
            # test_tgt = normalize_custom(torch.from_numpy(test_tgt), args.norm, None)
            if args.src:
                mean=trainer.src_mean
            elif args.tgt:
                mean=trainer.tgt_mean
            test_src,test_tgt=eval_prepocess_data((test_src,test_tgt),trainer,args.norm,mean)

            print (test_src.dtype)
            test_src_orig = deepcopy(test_src)
            test_tgt_orig = deepcopy(test_tgt)
            scores_pred = produce_cosine_list(test_src_orig, test_tgt_orig)
            rho = spearmanr(scores, scores_pred)[0]
            print (test_src.dtype)
            print('spearman rank for original embedding {0} is {1}'.format(testset, rho))
            if args.baseline:
                test_1_other, test_2_other, scores = produce_test_material(args.test_data_dir, testset,args.base_embed,
                                                        args.avg_flag, args.lg, args.context_num)
                test_src_other=test_1_other
                test_tgt_other=test_2_other
                test_src_other,test_tgt_other=eval_prepocess_data((test_src_other,test_tgt_other),trainer,args.norm,trainer.src_mean)
                
                test_src,test_tgt=add_baseline(test_src,test_src_other,args.baseline),add_baseline(test_tgt,test_tgt_other,args.baseline)
                    
            else:
                test_src, test_tgt = trainer.apply_model_en(test_src, test_tgt)

        scores_pred = produce_cosine_list(test_src, test_tgt)
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

def eval_prepocess_data(data,trainer,norm,mean=None):

    
    if type(data)==tuple:

        lens=[len(data_i) for data_i in data]
        data_all=np.concatenate(data)
        data_all=eval_prepocess_data(data_all,trainer,norm,mean)
        data_out=[]
        start=0
        for leni in lens:
            data_out.append(data_all[start:start+leni])
            start+=leni
        return data_out
    else:
        
        data = torch.from_numpy(data).type('torch.FloatTensor').to(trainer.device)
        
        normalize_embeddings(data, norm, mean)
        data=deepcopy(data)
    return data

def csimlex_test_data(args,base_embed):
    test_f,test_data_f=produce_context_simlex_data(args.test_data_dir,base_embed)

    test_data=np.load(test_data_f,allow_pickle=True)[()]
    test_0=[]
    test_1=[]
    for line in open(test_f):
        line=line.strip()
        text, loc1, loc2=line.strip().split('\t')[:3]
        # print (text,loc,score)
        try:
            embed1=test_data[produce_key(text)][int(loc1)]
            test_0.append(embed1)
            embed2=test_data[produce_key(text)][int(loc2)]
            test_1.append(embed2)
        except KeyError:
            print ('NOT FOUND: {0}'.format(text))


    return test_0,test_1

def output_csimlex(scores_pred,trainer,args):
    subtask1=['change']
    subtask2=['sim_context1\tsim_context2']
    test_dir=os.path.join(args.test_data_dir,'context_simlex')
    model_dir=os.path.join(test_dir,trainer.tag)
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



def eval_context_simlex(args,trainer):
    with torch.no_grad():
        data1,data2=csimlex_test_data(args,args.base_embed)

        if args.src:
            mean=trainer.src_mean
        elif args.tgt:
            mean=trainer.tgt_mean
        data_1,data_2=eval_prepocess_data((data1,data2),trainer,args.norm,mean)
        trainer.model_src.eval()
        if args.baseline:
            data1_other,data2_other=csimlex_test_data(args,args.base_embed)
            data_1_other,data_2_other=eval_prepocess_data((data1_other,data2_other),trainer,args.norm,trainer.src_mean)
            data_1,data_2=add_baseline(data_1,data_1_other,args.baseline),add_baseline(data_2,data_2_other,args.baseline)
            

        else:
            data_1, data_2 = trainer.apply_model_en(data_1, data_2)
        # data_1,data_2=eval_prepocess_data((data1,data2),trainer,CENTER,None)

        scores_pred = produce_cosine_list(data_1, data_2)
        output_csimlex(scores_pred,trainer,args)

def output_wic_test(test_pred,args,trainer):
    wic_test_dir=os.path.join(args.test_data_dir,'./WiC_dataset/test/'+trainer.tag)
    if not os.path.exists(wic_test_dir):
        os.mkdir(wic_test_dir)

    with open(os.path.join(wic_test_dir, 'output.txt'), 'w') as f:
        f.write('\n'.join(test_pred))
    # with ZipFile(os.path.join(wic_test_dir,'output.txt.zip'),'w') as f:
    #     f.write(os.path.join(wic_test_dir, 'output.txt'))

def output_wic_analysis(train_dev_scores,train_dev_golds,train_dev_texts,train_dev_texts_loc,train_dev_pred,args,trainer):
    wic_train_dev_dir=os.path.join(args.test_data_dir,'./WiC_dataset/train/'+trainer.tag)
    if not os.path.exists(wic_train_dev_dir):
        os.mkdir(wic_train_dev_dir)
    with open(os.path.join(wic_train_dev_dir, 'analysis.txt'), 'w') as f:
        for i in range(len(train_dev_texts)):
            text1,text2=train_dev_texts[i][0],train_dev_texts[i][1]
            loc1,loc2=train_dev_texts_loc[i][0],train_dev_texts_loc[i][1]
            w1,w2=text1.split()[int(loc1)],text2.split()[int(loc2)]
            f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n'.format(text1,text2,loc1,loc2,w1,w2,train_dev_golds[i],train_dev_scores[i],train_dev_pred[i]))
def wic_cat_baseline(args,trainer,train_1,train_2,dev_1,dev_2,test_1,test_2,crossling_flag):
    if crossling_flag:
        data_scores_others=produce_xlwic_data(args.test_data_dir,args.base_embed,args.src_lg+'2'+args.tgt_lg,args.lg)
    else:
        data_scores_others=produce_wic_data(args.test_data_dir,args.base_embed)
    train_1_other,train_2_other,_,_,_,_,_=data_scores_other['train']
    dev_1_other,dev_2_other,_,_,_,_,_=data_scores_other['dev']
    test_1_other,test_2_other,_,_,_,_,_=data_scores_other['test']
    train_1_other,train_2_other,dev_1_other,dev_2_other,test_1_other,test_2_other=eval_prepocess_data(( train_1_other,train_2_other,dev_1_other,dev_2_other,test_1_other,test_2_other),trainer,args.norm,trainer.src_mean)
    train_1,train_2=add_baseline(train_1,train_1_other,args.baseline),add_baseline(train_2,train_2_other,args.baseline)
    dev_1,dev_2=add_baseline(dev_1,dev_1_other,args.baseline),add_baseline(dev_2,dev_2_other,args.baseline)
    test_1,test_2=add_baseline(test_1,test_1_other,args.baseline),add_baseline(test_2,test_2_other,args.baseline)
    return train_1,train_2,dev_1,dev_2,test_1,test_2

def eval_wic(args,trainer,crossling_flag=False,align_flag=False):
    with torch.no_grad():
        if crossling_flag:
            data_scores=produce_xlwic_data(args.test_data_dir,args.base_embed,args.src_lg+'2'+args.tgt_lg,args.lg)
        else:
            data_scores=produce_wic_data(args.test_data_dir,args.base_embed)

        train_1,train_2,train_scores,train_1_text,train_2_text,train_1_text_loc,train_2_text_loc=data_scores['train']
        dev_1,dev_2,dev_scores,dev_1_text,dev_2_text,dev_1_text_loc,dev_2_text_loc=data_scores['dev']
        test_1,test_2,test_scores,test_1_text,test_2_text,test_1_text_loc,test_2_text_loc=data_scores['test']
        if args.src:
            mean=trainer.src_mean
        elif args.tgt:
            mean=trainer.tgt_mean
        train_1,train_2,dev_1,dev_2,test_1,test_2=eval_prepocess_data((train_1,train_2,dev_1,dev_2,test_1,test_2),trainer,args.norm,mean)
        if args.baseline:
            train_1,train_2,dev_1,dev_2,test_1,test_2=wic_cat_baseline(args,trainer,train_1,train_2,dev_1,dev_2,test_1,test_2,crossling_flag)
            # data_scores_others=produce_wic_data(args.test_data_dir,args.cat_baseline)
            # train_1_other,train_2_other,_,_,_,_,_=data_scores['train']
            # dev_1_other,dev_2_other,_,_,_,_,_=data_scores['dev']
            # test_1_other,test_2_other,_,_,_,_,_=data_scores['test']
            # train_1_other,train_2_other,dev_1_other,dev_2_other,test_1_other,test_2_other=eval_prepocess_data(( train_1_other,train_2_other,dev_1_other,dev_2_other,test_1_other,test_2_other),trainer,args.norm,trainer.src_mean)
            # train_1,train_2=torch.cat([train_1,train_1_other],dim=1),torch.cat([train_2,train_2_other],dim=1)
            # dev_1,dev_2=torch.cat([dev_1,dev_1_other],dim=1),torch.cat([dev_2,dev_2_other],dim=1)
            # test_1,test_2=torch.cat([test_1,test_1_other],dim=1),torch.cat([test_2,test_2_other],dim=1)
        else:
            print ('apply model ')
            if not align_flag:
                # pass
                train_1,train_2 = trainer.apply_model_en(train_1, train_2)
                dev_1,dev_2=trainer.apply_model_en(dev_1,dev_2)        
                test_1,test_2=trainer.apply_model_en(test_1,test_2)
                
            else:
                trainer.tag+='.aligned'
                train_1,train_2= trainer.apply_model(train_1,train_2)
                dev_1,dev_2=trainer.apply_model(dev_1,dev_2)        
                test_1,test_2=trainer.apply_model(test_1,test_2)
        
        
        train_scores_pred = produce_cosine_list(train_1, train_2)
        print ('==WIC RESULTS==: average cosine for training',sum(train_scores_pred)/len(train_scores_pred),len(train_scores_pred))
        dev_scores_pred = produce_cosine_list(dev_1, dev_2)

        print ('train_dev results for wic ')
       
        train_dev_scores_pred=train_scores_pred+dev_scores_pred
        train_dev_scores=train_scores+dev_scores
        train_dev_texts=list(zip(train_1_text,train_2_text))+list(zip(dev_1_text,dev_2_text))
        train_dev_texts_loc=list(zip(train_1_text_loc,train_2_text_loc))+list(zip(dev_1_text_loc,dev_2_text_loc))
      
        train_dev_thres, train_dev_pred = eval_wic_cosine(train_dev_scores_pred, train_dev_scores)

        print ('test results for wic with traindev thres')

        scores_pred = produce_cosine_list(test_1, test_2)
        thres, test_pred=eval_wic_cosine(scores_pred,test_scores,train_dev_thres)

        print ('train results for wic with traindev thres')
        _,_=eval_wic_cosine(train_scores_pred,train_scores,train_dev_thres)

        print ('dev results for wic with traindev thres')
        _,_=eval_wic_cosine(dev_scores_pred,dev_scores,train_dev_thres)

        output_wic_test(test_pred,args,trainer)
        output_wic_analysis(train_dev_scores_pred,train_dev_scores,train_dev_texts,train_dev_texts_loc,train_dev_pred,args,trainer)


def test2lst(test,lexsubcontext=None):
        w_list=[]
        with open(test,'r') as f:
            for line in f:
                fields=line.strip().split('\t')
                w=produce_key(fields[0])
                if lexsubcontext:
                    w_list.append((w,fields[1],fields[3]))
                else:
                    w_list.append(w)
        return sorted(list(set(w_list)))






