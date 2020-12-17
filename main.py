
import torch
import numpy as np
from helpers import *


def filter_one2many(align_lst):
    en_lst=[]
    zh_lst=[]
    avoid_en_list=[]
    avoid_zh_list=[]

    for pair in align_lst:
        zh_index, en_index = pair.split('-')
        if zh_index in zh_lst:
            avoid_zh_list.append(zh_index)
        else:
            zh_lst.append(zh_index)
        if en_index in en_lst:
            avoid_en_list.append(en_index)
        else:
            en_lst.append(en_index)

    filtered_lst=[pair for pair in align_lst if pair.split('-')[0] not in avoid_zh_list and pair.split('-')[1] not in avoid_en_list]
    return filtered_lst





def retrieve_type_emb(zh_line,en_line,align_line,zh_data_dict,en_data_dict,wordpair2avgemb,en2zh_muse):
    zh_line_data_per_line=[]
    en_line_data_per_line=[]
    zh_line_data_all=[]
    en_line_data_all=[]
    for pair in filter_one2many(align_line.strip().split()):
        zh_index, en_index = pair.split('-')
        zh_w=zh_line.split()[int(zh_index)]
        en_w=en_line.split()[int(en_index)]
        zh_w = produce_key(zh_w.lower())
        en_w = produce_key(en_w.lower())

        if (not en2zh_muse) or (en_w in en2zh_muse and zh_w in en2zh_muse[en_w]['zh']):
            # print ('type:', zh_w,en_w)
            if zh_w in zh_data_dict:
                zh_line_data_all.append(zh_data_dict[zh_w][0])
            if en_w in en_data_dict:
                en_line_data_all.append(en_data_dict[en_w][0])

            if (zh_w, en_w) not in wordpair2avgemb:
                wordpair2avgemb[(zh_w, en_w)] = True
                if zh_w in zh_data_dict and en_w in en_data_dict:
                    zh_line_data = zh_data_dict[zh_w]
                    en_line_data = en_data_dict[en_w]
                    zh_line_data_per_line.append(zh_line_data)
                    en_line_data_per_line.append(en_line_data)
        # else:
        #     print ('NOT IN DICT: {0} {1}'.format(zh_w,en_w))

    return zh_line_data_per_line,en_line_data_per_line,zh_line_data_all,en_line_data_all

def retrieve_sentemb(zh_line,en_line,zh_data_f_lst,en_data_f_lst):
    zh_data=extract_emb(zh_data_f_lst,'[CLS] '+zh_line)
    en_data=extract_emb(en_data_f_lst,'[CLS] '+en_line)
    if type(zh_data)==type(None) or type(en_data)==type(None):
        return None, None
    return [zh_data],[en_data]

def construct_types_dict(zh_data_f_lst):
    zh_data_dict={}
    for zh_data_f in zh_data_f_lst:
        for key in zh_data_f:
            zh_data_dict[key]=zh_data_f[key][:]
    return zh_data_dict

def construct_types_dict_from_vocabfile(en_vocab):
    en_vocab_dict={}
    with open(en_vocab, 'r') as en_vocab_f:
        for line in en_vocab_f:
            w=line.strip().split('\t')[0]
            en_vocab_dict[w]=True
    return en_vocab_dict

def cluster_wsd_lg(en2zh_muse,wordpair2avgemb,lg,zh_w,en_w):
    zh=lg.split('2')[1]
    en_emb = []
    contextall_flag=False
    if 'wsd' in en2zh_muse[en_w]:
        wsd_lst = en2zh_muse[en_w]['wsd']
    else:
        wsd_lst = [0] * len(en2zh_muse.get(en_w).get(zh))
    wsd_index = wsd_lst[en2zh_muse.get(en_w).get(zh).index(zh_w)]
    print('clustering ', en_w, 'wordpair',en_w, zh_w)
    for i, zh_w_loop in enumerate(en2zh_muse.get(en_w).get(zh)):
        if lg =='en2zh':
            wordpair=(zh_w_loop, en_w)
        elif lg== 'zh2en':
            wordpair=(en_w,zh_w_loop)

        if wordpair in wordpair2avgemb:

            if wsd_lst[i] == wsd_index:
                print (en_w,zh_w_loop)
                if wordpair2avgemb.get(wordpair).get(lg)[0]!='contextall':
                    en_emb += wordpair2avgemb.get(wordpair).get(lg)
                else:
                    contextall_flag=True
    if not en_emb and contextall_flag: # wsd must have something. Fall back to contextall
        src,tgt=retrieve_source_tgt_fromwp(wordpair,lg)
        en_emb = wordpair2avgemb.get(src).get(lg)

    return en_emb


def cluster_wsd(en2zh_muse,zh2en_muse,wordpair,wordpair2avgemb):
    zh_w,en_w=wordpair
    en_emb=cluster_wsd_lg(en2zh_muse,wordpair2avgemb,'en2zh',zh_w,en_w)
    zh_emb=cluster_wsd_lg(zh2en_muse,wordpair2avgemb,'zh2en',en_w,zh_w)
    # en_emb=[]
    en_wsd='no'
    if en_w in en2zh_muse:
        if 'wsd' in en2zh_muse[en_w]:
            en_wsd= 'wsd'
    zh_wsd='no'
    if zh_w in zh2en_muse:
        if 'wsd' in zh2en_muse[zh_w]:
          zh_wsd='wsd'

    #     wsd_lst=en2zh_muse[en_w]['wsd']
    # else:
    #     wsd_lst=[0]*len(en2zh_muse[en_w]['zh'])
    # wsd_index=wsd_lst[en2zh_muse[en_w]['zh'].index(zh_w)]
    # for i,zh_w_loop in enumerate(en2zh_muse[en_w]['zh']):
    #     if (zh_w_loop, en_w) in wordpair2avgemb:
    #         if wsd_lst[i]==wsd_index:
    #             en_emb+=wordpair2avgemb[(zh_w_loop, en_w)]['en2zh']
    #
    # zh_emb = []
    # if 'wsd' in zh2en_muse[zh_w]:
    #     wsd_lst = zh2en_muse[zh_w]['wsd']
    # else:
    #     wsd_lst = [0] * len(zh2en_muse[zh_w]['en'])
    # wsd_index = wsd_lst[en2zh_muse[zh_w]['en'].index(en_w)]
    # for i, zh_w_loop in enumerate(en2zh_muse[en_w]['zh']):
    #     if (zh_w_loop, en_w) in wordpair2avgemb:
    #         if wsd_lst[i] == wsd_index:
    #             en_emb += wordpair2avgemb[(zh_w_loop, en_w)]['en2zh']
    print ('wordpairs:',en_w,zh_w,en_wsd,zh_wsd,len(en_emb),len(zh_emb))
    # print ('zh_emb, en_emb',len(zh_emb),len(en_emb))
    return zh_emb,en_emb,zh_wsd,en_wsd


def cos_avg(zh_data_wp_token, zh_data_wp_type):
    cos=cosine_similarity(zh_data_wp_token,zh_data_wp_type)[0]+1
    zh_data_weighted=sum((np.array(zh_data_wp_type).T*np.array(cos)).T)/sum(cos)
    return zh_data_weighted

def filter_rare(wsd_lst,trans_lst,thresh=0.2):
    wsd2freq=defaultdict(float)
    trans_freq_all=float(sum(trans_lst))
    rare_wsd=[]
    for i,wsd in enumerate(wsd_lst):
        wsd2freq[wsd]+=trans_lst[i]
    max_freq=list(wsd2freq.values())[0]

    min_freq=list(wsd2freq.values())[0]
    for wsd in wsd2freq:
        if wsd2freq[wsd]/trans_freq_all<thresh:
            rare_wsd.append(wsd)
        if wsd2freq[wsd]>=max_freq:
            max_freq=wsd2freq[wsd]
            max_freq_wsd=wsd
        if wsd2freq[wsd]<=min_freq:
            min_freq=wsd2freq[wsd]
            min_freq_wsd=wsd
    return rare_wsd, max_freq_wsd,min_freq_wsd

def count_wsd(en2zh_muse_data,lg,percent,filter_flag=None):
    count_wsd = 0
    count_rare = 0
    count_balance = 0
    en2zh_out = defaultdict(lambda: defaultdict(list))
    count_all=0
    wsd_found_f='wsd_found_cwn'
    with open(wsd_found_f,'a+') as wsd_found_f:
        for en_w in en2zh_muse_data:
            count_all+=len(en2zh_muse_data[en_w][lg])
            if 'wsd' not in en2zh_muse_data[en_w]:
                en2zh_out[en_w] = en2zh_muse_data[en_w]
            else:
                if len(set(en2zh_muse_data[en_w]['wsd'])) <= 1:
                    en2zh_out[en_w] = en2zh_muse_data[en_w]
                else:
                    rare_wsd, max_freq_wsd, min_freq_wsd = filter_rare(en2zh_muse_data[en_w]['wsd'],
                                                                       en2zh_muse_data[en_w]['trans_freq'])
                    print(en_w, en2zh_muse_data[en_w][lg], rare_wsd, max_freq_wsd, min_freq_wsd)

                    for i, zh_w in enumerate(en2zh_muse_data[en_w][lg]):
                        count_wsd += 1
                        if rare_wsd == []:
                            count_balance += 1
                        else:
                            if en2zh_muse_data[en_w]['wsd'][i] in rare_wsd:
                                count_rare += 1

                        if lg=='zh':
                            wsd_found_f.write('{0}\t{1}\t{2}\t{3}\n'.format(en_w, zh_w,'', ''))
                        elif lg=='en':
                            wsd_found_f.write('{0}\t{1}\t{2}\t{3}\n'.format(zh_w, en_w, '', ''))
                        print('{3}\ttrans freq from\t {0}\t{1}\t{2}\t{4}'.format(en_w, zh_w,
                                                                                 en2zh_muse_data[en_w]['trans_freq'][i], lg,
                                                                                 en2zh_muse_data[en_w]['wsd'][i]))
                        if filter_flag == 'max':  # retain only the max:
                            if en2zh_muse_data[en_w]['wsd'][i] == max_freq_wsd:
                                print('ADD:{3}\ttrans freq from\t {0}\t{1}\t{2}\t{4}'.format(en_w, zh_w,
                                                                                             en2zh_muse_data[en_w][
                                                                                                 'trans_freq'][i], lg,
                                                                                             en2zh_muse_data[en_w]['wsd'][
                                                                                                 i]))
                                en2zh_out[en_w][lg].append(zh_w)
                        elif filter_flag == 'min':
                            if en2zh_muse_data[en_w]['wsd'][i] == min_freq_wsd:
                                print('ADD:{3}\ttrans freq from\t {0}\t{1}\t{2}\t{4}'.format(en_w, zh_w,
                                                                                             en2zh_muse_data[en_w][
                                                                                                 'trans_freq'][i], lg,
                                                                                             en2zh_muse_data[en_w]['wsd'][
                                                                                                 i]))
                                en2zh_out[en_w][lg].append(zh_w)
                        else:
                            print('ADD:{3}\ttrans freq from\t {0}\t{1}\t{2}\t{4}'.format(en_w, zh_w, en2zh_muse_data[en_w][
                                'trans_freq'][i], lg,
                                                                                         en2zh_muse_data[en_w]['wsd'][i]))
                            en2zh_out[en_w][lg].append(zh_w)

        print('wordpairs total:',count_all,'wsd total: {0}'.format(count_wsd), 'wsd rare: {0}'.format(count_rare),
              'count_balance: {0}'.format(count_balance))
        # print ('wsd perc', count_wsd/float(count_all))
        if percent:
            en2zh_out=change_wsd_per(en2zh_muse_data,count_wsd,lg,percent)
    return en2zh_out

def change_wsd_per(en2zh_muse_out,count_wsd,lg,percent):
    count_mono_exp=int(count_wsd/percent)-count_wsd
    count_mono_current=0
    en2zh_out_exp={}
    for en_w in en2zh_muse_out:
        if 'wsd' not in en2zh_muse_out[en_w] or len(set(en2zh_muse_out[en_w]['wsd'])) <= 1:
           count_mono_current+=len(en2zh_muse_out[en_w][lg])
           if count_mono_current <= count_mono_exp:
               en2zh_out_exp[en_w] = en2zh_muse_out[en_w]
        else:
            en2zh_out_exp[en_w]=en2zh_muse_out[en_w]
    print ('mono word pairs:',count_mono_current, 'wsd word pairs', count_wsd)
    return en2zh_out_exp


def update_en2zh_muse(en2zh_muse,en2zh_muse_data):
    for w in list(en2zh_muse.keys()):
        if w in en2zh_muse_data:
            en2zh_muse[w]=en2zh_muse_data[w]
        if w not in en2zh_muse_data:
            del en2zh_muse[w]
    assert en2zh_muse==en2zh_muse_data

def add_sense_freq(en2zh_muse, wordpair2avgemb,lg,freq_thres):
    # if not filter_flag:
    #     return en2zh_muse
    en2zh_muse_data=defaultdict(lambda: defaultdict(list))
    for en_w in en2zh_muse:
        for i,zh_w in enumerate(en2zh_muse.get(en_w).get(lg)):
                if lg=='zh':
                    wordpair=(zh_w,en_w)
                    lg_dir='en2zh'
                elif lg=='en':
                    wordpair = (en_w, zh_w)
                    lg_dir='zh2en'
                if wordpair in wordpair2avgemb and lg_dir in wordpair2avgemb.get(wordpair):


                    en_w_len=len(wordpair2avgemb.get(wordpair).get(lg_dir))
                    # print ('{3}\ttrans freq from\t {0}\t{1}\t{2}\t{4|'.format(en_w, zh_w, zh_w_len,lg,en2zh_muse[en_w]['wsd'][i]))
                    if en_w_len>=freq_thres or wordpair2avgemb.get(wordpair).get(lg_dir)[0]=='contextall': # at least 1 occurence
                        en2zh_muse_data[en_w][lg].append(zh_w)
                        if 'wsd' in en2zh_muse[en_w]:
                            en2zh_muse_data[en_w]['wsd'].append(en2zh_muse.get(en_w).get('wsd')[i])
                        en2zh_muse_data[en_w]['trans_freq'].append(en_w_len)

    # en2zh_out=count_wsd(en2zh_muse_data, lg, filter_flag)
    update_en2zh_muse(en2zh_muse, en2zh_muse_data)
    return en2zh_muse_data

def produce_clusteremb_trans(zh_ws,wsd,wordpair2avgemb,lg):
    lg=lg.split('2')[1]+'2'+lg.split('2')[0]
    cluster2emb = defaultdict(list)
    # cluster2trans=defaultdict(list)


    for i, zh_w in enumerate(zh_ws):
        # if lg == 'en2zh':
        #     wordpair = (zh_w, en_w)
        # elif lg == 'zh2en':
        #     wordpair = (en_w, zh_w)
        if zh_w in wordpair2avgemb and lg in wordpair2avgemb.get(zh_w):
            tran_emb=sum(wordpair2avgemb.get(zh_w).get(lg))/len(wordpair2avgemb.get(zh_w).get(lg))
            cluster2emb[wsd[i]].append(tran_emb)
            # cluster2trans.append()
    return cluster2emb


def produce_clusteremb(en_w,zh_ws,wsd,wordpair2avgemb,lg):
    cluster2emb = defaultdict(list)

    for i, zh_w in enumerate(zh_ws):
            if lg=='en2zh':
                wordpair = (zh_w, en_w)
            elif lg=='zh2en':
                wordpair= (en_w,zh_w)
            if wordpair in wordpair2avgemb and lg in wordpair2avgemb.get(wordpair):
                if wordpair2avgemb.get(wordpair).get(lg)[0]!='contextall':
                    cluster2emb[wsd[i]] += wordpair2avgemb.get(wordpair).get(lg)
    return cluster2emb

def produce_embx_labely(cluster2emb,cluster_compare):
    embs=np.vstack([cluster2emb[cluster] for cluster in cluster_compare])
    labels=[]
    for cluster in cluster_compare:
        labels+=[cluster]*len(cluster2emb[cluster])
    return embs,labels

def search_for_mincluster(cluster2new_cluster,cluster):

    if cluster not in cluster2new_cluster:
        return [cluster]
    else:
        l=[]
        for cluster_nest in cluster2new_cluster[cluster]:
            l+=search_for_mincluster(cluster2new_cluster,cluster_nest)

        return l

def update_wsd(cluster2new_cluster,wsd_lst):
    wsd_lst_new=[]
    update_flag=False
    for wsd in wsd_lst:
        wsd_new=min(search_for_mincluster(cluster2new_cluster,wsd))
        wsd_lst_new.append(wsd_new)
    if wsd_lst_new!=wsd_lst:
        update_flag=True
    return wsd_lst_new,update_flag

def retrieve_source_tgt_fromwp(wp,lg):
    if lg=='en2zh':
        source=wp[1]
        tgt=wp[0]
    elif lg=='zh2en':
        source=wp[0]
        tgt=wp[1]
    return source,tgt

def form_wp_from_lg(src,tgt,lg):
    if lg=='en2zh':
        wp=(tgt,src)
    elif lg =='zh2en':
        wp=(src,tgt)
    return wp
def fix_semisupervised_cluster(en2zh_muse,zh2en_muse, wordpair2avgemb):
    lg2dict={'en2zh':en2zh_muse,'zh2en':zh2en_muse}
    for wp in wordpair2avgemb:

        if type(wp)==tuple:
            for lg in wordpair2avgemb[wp]:
                if lg in ['en2zh','zh2en']:

                    src, tgt = retrieve_source_tgt_fromwp(wp, lg)
                    tgt_lg = lg.split('2')[1]

                    if wordpair2avgemb.get(wp).get(lg)[0]=='contextall':
                        if src in lg2dict.get(lg) and 'wsd' in lg2dict.get(lg).get(src) and tgt in lg2dict.get(lg).get(src).get(tgt_lg):
                            wp_solved_flag = False
                            wsd=lg2dict.get(lg).get(src).get(lg).get(src).get('wsd')
                            tgt_words=lg2dict.get(lg).get(src).get(tgt_lg)
                            tgt_w_i=tgt_words.index(tgt)
                            tgt_wsd=wsd[tgt_w_i]
                            tgt_words_samewsd=[tgt_words[i] for i,wsd_i in enumerate(wsd) if wsd_i==tgt_wsd]
                            for tgt_w_candi in tgt_words_samewsd:
                                wp_candi=form_wp_from_lg(src,tgt_w_candi,lg)
                                if wordpair2avgemb.get(wp_candi).get(lg)[0]!='contextall':
                                    wordpair2avgemb[wp][lg]=wordpair2avgemb.get(wp_candi).get(lg)
                                    wp_solved_flag=True
                            if not wp_solved_flag:
                                lg2dict.get[lg][src]['wsd'][tgt_w_i]=min(0,min(lg2dict.get[lg][src]['wsd']))-1



def update_sil(cluster_compares,cluster2new_cluster,cluster2emb,thres):
    for cluster_compare in cluster_compares:
        embs, labels = produce_embx_labely(cluster2emb, cluster_compare)
        sil = silhouette_score(embs, labels)
        if sil < thres:
            cluster2new_cluster[max(cluster_compare)].append(min(cluster_compare))
        print('sil wsd', sil, cluster_compare)

def update_trans(cluster_compares,cluster2new_cluster,cluster2transemb,word2maxcos,thres):
    for cluster_compare in cluster_compares:
        cos_matrix=cosine_similarity(cluster2transemb[cluster_compare[0]],cluster2transemb[cluster_compare[1]])
        cos_matrix=cos_matrix.reshape(1,cos_matrix.shape[0]*cos_matrix.shape[1])[0]
        cos_mean=sum(cos_matrix)/len(cos_matrix)
        if cos_mean>thres:
            cluster2new_cluster[max(cluster_compare)].append(min(cluster_compare))
        print ('cos wsd', cos_mean,cluster_compare)


def update_wsd_perwp_bysiltrans(en_w,en2zh_muse,wordpair2avgemb,zh,lg,word2maxcos,wsd_thres):

    en2zh_muse_enw=en2zh_muse[en_w]

    if 'wsd' in en2zh_muse_enw:
        print('sil update wordpairs', en_w, en2zh_muse_enw[zh],en2zh_muse_enw['wsd'])
        update_flag=True
        while update_flag:
            cluster2new_cluster = defaultdict(list)
            wsd = en2zh_muse_enw['wsd']
            print ('sil update before', wsd)
            zh_ws = en2zh_muse_enw[zh]
            cluster2transemb=produce_clusteremb_trans(zh_ws,wsd,wordpair2avgemb,lg)
            cluster_compares_trans=permutate(list(set(cluster2transemb.keys())))
            update_trans(cluster_compares_trans,cluster2new_cluster,cluster2transemb,word2maxcos,thres=wsd_thres)
            # cluster2emb = produce_clusteremb(en_w, zh_ws, wsd, wordpair2avgemb, lg)
            # cluster_compares = permutate(list(set(cluster2emb.keys())))
            # update_sil(cluster_compares,cluster2new_cluster,cluster2emb,thres=0.15)


            # for cluster_compare in cluster_compares:
            #     embs,labels=produce_embx_labely(cluster2emb,cluster_compare)
            #     sil=silhouette_score(embs,labels)
            #     if sil<0.15:
            #             cluster2new_cluster[max(cluster_compare)].append(min(cluster_compare))
            #     print ('sil wsd',sil, cluster_compare)

            print ('sil wsd cluster2cluster_new',cluster2new_cluster)
            en2zh_muse_enw['wsd'],update_flag=update_wsd(cluster2new_cluster,wsd)
            print ('sil update after', en2zh_muse_enw['wsd'])


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

def produce_transvocab_cos_matrix(wordpair2avgemb,lg):
    lg=lg.split('2')[1] + '2' + lg.split('2')[0]
    vocab_emb=[]
    index2word={}
    word2index={}
    count=0
    for w in wordpair2avgemb:
        if type(w)!=tuple and lg in wordpair2avgemb.get(w):
            clusterall_emb=wordpair2avgemb.get(w).get(lg)
            clusterall_emb=sum(clusterall_emb)/len(clusterall_emb)
            vocab_emb.append(clusterall_emb)
            index2word[count]=w
            word2index[w]=count

            count+=1

    cos_matrix=cosine_similarity(vocab_emb,vocab_emb)
    cos_matrix=skip_diag_strided(cos_matrix)
    cos_max=np.max(cos_matrix,axis=1)
    word2maxcos={index2word[i]:cos for i,cos in enumerate(cos_max)}
    return word2maxcos

def update_wsd_by_sil_trans(wordpair2avgemb,en2zh_muse,lg,wsd_thres):

    zh = lg.split('2')[1]
    # word2maxcos=produce_transvocab_cos_matrix(wordpair2avgemb)
    word2maxcos=None
    for en_w in en2zh_muse:
        update_wsd_perwp_bysiltrans(en_w, en2zh_muse, wordpair2avgemb, zh, lg,word2maxcos,wsd_thres)



def update_clusterall(en2zh_muse,lg,wordpair2avgemb):
    zh=lg.split('2')[1]
    for en_w in en2zh_muse:
        no_update = False

        word_clusterall=[]
        goldtrans=[]
        for zh_w in en2zh_muse[en_w][zh]:
            if lg=='zh2en':
                wordpair=(en_w,zh_w)
            elif lg=='en2zh':
                wordpair=(zh_w,en_w)
            if wordpair2avgemb.get(wordpair).get(lg)[0]=='contextall': # do not update the source word clusterall
                no_update=True
                break
            word_clusterall+=wordpair2avgemb[wordpair][lg]
            goldtrans+=[zh_w]*len(wordpair2avgemb[wordpair][lg])
            if len(wordpair2avgemb[wordpair][lg])<5:
                print ('ATTENTION, wp smaller than freq thres')
        if not no_update:
            wordpair2avgemb[en_w][lg]=word_clusterall
            wordpair2avgemb[en_w][lg+'-transgold'] = goldtrans
            if len(word_clusterall)==0:
                del wordpair2avgemb[en_w][lg]
                del wordpair2avgemb[en_w][lg+'-transgold']

def extract_wa_means(wordpair2avgemb,word,lg):
    # if lg=='zh':
    #     word=
    wa_means=[]
    for wp in wordpair2avgemb:
        if type(wp)==tuple:
            zh_w,en_w=wp
            if lg=='zh2en':
                w2focus=zh_w
            elif lg=='en2zh':
                w2focus=en_w
            if w2focus==word and lg in wordpair2avgemb[wp]:
                wa_means+=[sum(wordpair2avgemb[wp][lg])/len(wordpair2avgemb[wp][lg])]*len(wordpair2avgemb[wp][lg])
    return wa_means


def fix_clusterwa_aftersemi(en_wp_all,wordpair2avgemb,wordpair,lg):
    en_wp_all_out=en_wp_all
    if en_wp_all[0]=='contextall':
        src,tgt=retrieve_source_tgt_fromwp(wordpair,lg)
        en_wp_all_out=wordpair2avgemb.get(src).get(lg)
    return en_wp_all_out


def add_labels(len_zh_data,len_en_data,zh_data_labels,en_data_labels,wordpair,wordpair2avgemb):
    if len_zh_data == len(wordpair2avgemb[wordpair[0]]['zh2en']):
        zh_data_labels.append(wordpair[0])
    else:
        zh_data_labels.append('.'.join(wordpair))
    if len_en_data == len(wordpair2avgemb[wordpair[1]]['en2zh']):
        en_data_labels.append(wordpair[1])
    else:
        en_data_labels.append('.'.join([wordpair[1],wordpair[0]]))

def cluster_results(en_clustercenters,zh_clustercenters,en_data_labels,zh_data_labels,en_data,zh_data,num_cluster):
    
    cluster_decision=True
    unique_zh,counts_zh=np.unique(zh_data_labels,return_counts=True)
    clusters_corresponds=[]
    clusters_embed_corresponds=[]
    cluster_cos_en=cosine_similarity([en_clustercenters[0]],[en_clustercenters[1]])
    cluster_cos_zh=cosine_similarity([zh_clustercenters[0]],[zh_clustercenters[1]])

    # print ('cluster en {0} and en {1} cosine similarity {2}'.format(0,1,cluster_cos_en))
    # print ('cluster zh {0} and zh {1} cosine similarity {2}'.format(0,1,cluster_cos_zh))

    for en_i in range(num_cluster):
        en_indexes=np.where(en_data_labels==en_i)
        zh_counterpart=zh_data_labels[en_indexes]
        unique, counts = np.unique(zh_counterpart, return_counts=True)
        max_i=np.argmax(counts)
        zh_correspond_cluster=unique[max_i]
        precision=float(counts[max_i]/len(en_indexes[0]))
        recall=float(counts[max_i]/counts_zh[np.where(unique_zh==zh_correspond_cluster)])
        
        if recall<=0.8 or precision<=0.8:
            cluster_decision=False
            return None, None
        
        else:
            print (recall,precision)
            
            en_data_cluster=list(np.array(en_data)[en_indexes])
            zh_data_cluster=list(np.array(zh_data)[np.where(zh_data_labels==zh_correspond_cluster)])
            if len(en_data_cluster)<5 or len(zh_data_cluster)<5:
                return None,None
            clusters_corresponds.append((en_i,zh_correspond_cluster))
            clusters_embed_corresponds.append((en_data_cluster,zh_data_cluster))

    return clusters_corresponds,clusters_embed_corresponds
    
def ssd(A,B):
  dif = A.ravel() - B.ravel()
  return np.dot( dif, dif )

def unsup_cluster(wordpair2avgemb,num_cluster):
    wordpair2avgemb_unsup=defaultdict(lambda: defaultdict(list))
    for wp in wordpair2avgemb:
        if type(wp)==tuple:

            zh_data_wp_all = wordpair2avgemb[wp[0]]['zh2en']
            en_data_wp_all = wordpair2avgemb[wp[1]]['en2zh']
            if len(zh_data_wp_all)<10:
                continue
            print (wp)
            if not wp[0].isalpha():
                continue
            kmeans = KMeans(n_clusters=num_cluster).fit(zh_data_wp_all)
            zh_clustercenters=kmeans.cluster_centers_
            tgt_inertia=kmeans.inertia_/len(zh_data_wp_all)
            tgt_all_inertia=np.sum((np.array(zh_data_wp_all) - np.mean(np.array(zh_data_wp_all),axis=0)) ** 2, dtype=np.float64)/len(zh_data_wp_all)
            tgt_inertia_ratio=tgt_inertia/tgt_all_inertia
            zh_data_labels=np.array(kmeans.labels_)
            kmeans = KMeans(n_clusters=num_cluster).fit(en_data_wp_all)
            en_clustercenters=kmeans.cluster_centers_
            src_inertia=kmeans.inertia_/len(en_data_wp_all)
            src_inertia_all=np.sum((np.array(en_data_wp_all) - np.mean(np.array(en_data_wp_all),axis=0)) ** 2, dtype=np.float64)/len(en_data_wp_all)
            src_inertia_ratio=src_inertia/src_inertia_all
            en_data_labels=np.array(kmeans.labels_)
            print ('inertia',tgt_inertia_ratio,src_inertia_ratio)
            # if tgt_inertia_ratio>0.6 or src_inertia_ratio>0.6:
            #     continue
            
            clusters_corresponds,clusters_embed_corresponds=cluster_results(en_clustercenters,zh_clustercenters,en_data_labels,zh_data_labels,en_data_wp_all,zh_data_wp_all,num_cluster)
            if type(clusters_corresponds)==type(None):
                continue
            for i,cluster_correspond in enumerate(clusters_corresponds):
                print ('found disambiguated',wp,clusters_corresponds)
                en_cluster,zh_cluster=clusters_corresponds
                en_clustered_data,zh_clustered_data=clusters_embed_corresponds[i]
                wordpair2avgemb_unsup[wp[0]+'|'+str(zh_cluster)]['zh2en']=zh_clustered_data
                wordpair2avgemb_unsup[wp[0]+'|'+str(zh_cluster)]['2encluster']=en_cluster
                wordpair2avgemb_unsup[wp[1]+'|'+str(en_cluster)]['en2zh']=en_clustered_data
                wordpair2avgemb_unsup[wp[1]+'|'+str(en_cluster)]['2zhcluster']=zh_cluster
    wordpair2avgemb.update(wordpair2avgemb_unsup)




def clusterall(mean_size,wordpair2avgemb,wordpair,zh_data_labels,en_data_labels):
    num = mean_size
    zh_data_wp_all = wordpair2avgemb[wordpair[0]]['zh2en']
    en_data_wp_all = wordpair2avgemb[wordpair[1]]['en2zh']

    if len(zh_data_wp_all) > num:
        random.seed(0)
        zh_data_wp_all=random.sample(zh_data_wp_all,num)
    if len(en_data_wp_all) > num:
        random.seed(0)
        en_data_wp_all = random.sample(en_data_wp_all, num)

    zh_data_wp=[sum(zh_data_wp_all)/len(zh_data_wp_all)]
    en_data_wp=[sum(en_data_wp_all)/len(en_data_wp_all)]
    zh_data_labels.append(wordpair[0])
    en_data_labels.append(wordpair[1])
    return zh_data_wp,en_data_wp
        
def produce_avgemb_cluster(mean_size,wsd_thres,freq_thres,wordpair2avgemb,en2zh_muse,zh2en_muse,cluster_flag,zh_data,en_data,percent,src=None,trainer=None):


    num_cluster=2
    en2zh_muse=add_sense_freq(en2zh_muse,  wordpair2avgemb,'zh',freq_thres)
    zh2en_muse=add_sense_freq(zh2en_muse,  wordpair2avgemb,'en',freq_thres)
    if cluster_flag in ['cluster_wsd','cluster_wsd_random']:
        update_wsd_by_sil_trans(wordpair2avgemb, en2zh_muse, 'en2zh',wsd_thres)
        update_wsd_by_sil_trans(wordpair2avgemb, zh2en_muse, 'zh2en',wsd_thres)
    update_clusterall(en2zh_muse,'en2zh',wordpair2avgemb)
    update_clusterall(zh2en_muse,'zh2en',wordpair2avgemb)

    en2zh_out = count_wsd(en2zh_muse, 'zh',percent)
    zh2en_out = count_wsd(zh2en_muse,'en',percent)



    # if cluster_flag=='token':
    #     return zh_data,en_data
    zh_data=[]
    en_data=[]
    zh_data_labels=[]
    en_data_labels=[]
    sample_times = 3
    # average context embeddings per word
    # for wordpair in wordpair2avgemb:
    #     if type(wordpair) != tuple:
    #         if 'en2zh' in  wordpair2avgemb[wordpair]:
    #             wordpair2avgemb[wordpair]['en2zh']=sum(wordpair2avgemb[wordpair]['en2zh'])/len(wordpair2avgemb[wordpair]['en2zh'])
    #         if 'zh2en' in wordpair2avgemb[wordpair]:
    #             wordpair2avgemb[wordpair]['zh2en']=sum(wordpair2avgemb[wordpair]['zh2en'])/len(wordpair2avgemb[wordpair]['zh2en'])
    if 'unsup' in cluster_flag:
        unsup_cluster(wordpair2avgemb,num_cluster)
    wordkeys=[wordpair for wordpair in wordpair2avgemb if type(wordpair) is tuple]

    for wordpair in sorted(wordkeys):
        # if type(wordpair) is tuple:
            if (en2zh_muse=={}) or (wordpair[0] in zh2en_out and wordpair[1] in zh2en_out[wordpair[0]]['en'] and wordpair[1] in en2zh_out and wordpair[0] in en2zh_out[wordpair[1]]['zh']):
                ## word sense distr:

                # random data distr wsd
                # zh_emb, en_emb=cluster_wsd(en2zh_muse, zh2en_muse, wordpair, wordpair2avgemb)
                # zh_data_wp=random.sample(zh_emb, wordpair2avgemb[wordpair]['count'])
                # en_data_wp=random.sample(en_emb, wordpair2avgemb[wordpair]['count'])


                # random data distr wa
                # random.shuffle(wordpair2avgemb[wordpair]['zh2en'])
                # random.shuffle(wordpair2avgemb[wordpair]['en2zh'])
                # zh_data_wp=wordpair2avgemb[wordpair]['zh2en']
                # en_data_wp=wordpair2avgemb[wordpair]['en2zh']

                # random data distr word
                # zh_data_wp= random.sample(wordpair2avgemb[wordpair[0]]['zh2en'], wordpair2avgemb[wordpair]['count'])
                # en_data_wp=random.sample(wordpair2avgemb[wordpair[1]]['en2zh'], wordpair2avgemb[wordpair]['count'])

                # cluster wa one2one

                if cluster_flag=='cluster_wa':
                    num = mean_size
                    zh_data_wp_all = wordpair2avgemb[wordpair]['zh2en']
                    en_data_wp_all = wordpair2avgemb[wordpair]['en2zh']

                    if len(zh_data_wp_all) > num:
                        random.seed(0)
                        zh_data_wp_all = random.sample(zh_data_wp_all, num)
                    if len(en_data_wp_all) > num:
                        random.seed(0)
                        en_data_wp_all = random.sample(en_data_wp_all, num)
                    zh_data_wp_all = fix_clusterwa_aftersemi(zh_data_wp_all, wordpair2avgemb, wordpair, 'zh2en')
                    en_data_wp_all = fix_clusterwa_aftersemi(en_data_wp_all, wordpair2avgemb, wordpair, 'en2zh')
                    zh_data_wp = [sum(zh_data_wp_all) / len(zh_data_wp_all)]
                    en_data_wp = [sum(en_data_wp_all) / len(en_data_wp_all)]
                    add_labels(len(zh_data_wp_all), len(en_data_wp_all), zh_data_labels, en_data_labels, wordpair,
                               wordpair2avgemb)
                    # zh_data_wp=[sum(wordpair2avgemb[wordpair]['zh2en'])/len(wordpair2avgemb[wordpair]['zh2en'])]
                    # en_data_wp = [sum(wordpair2avgemb[wordpair]['en2zh']) / len(wordpair2avgemb[wordpair]['en2zh'])]

                

                elif cluster_flag=='cluster_wa_all':
                    zh_data_wp_all = wordpair2avgemb[wordpair]['zh2en']
                    en_data_wp_all = wordpair2avgemb[wordpair]['en2zh']
                    zh_data_wp_clusterall_all = wordpair2avgemb[wordpair[0]]['zh2en']
                    en_data_wp_clusterall_all = wordpair2avgemb[wordpair[1]]['en2zh']

                    zh_data_wp = [sum(zh_data_wp_all) / len(zh_data_wp_all)]
                    en_data_wp = [sum(en_data_wp_all) / len(en_data_wp_all)]
                    if len(zh_data_wp_all)!= len(zh_data_wp_clusterall_all):
                        zh_data_wp.append(sum(zh_data_wp_clusterall_all)/len(zh_data_wp_clusterall_all))
                        if len(en_data_wp_all)!= len(en_data_wp_clusterall_all):
                            en_data_wp.append(sum(en_data_wp_clusterall_all)/len(en_data_wp_clusterall_all))
                        else:
                            en_data_wp+=en_data_wp
                    else:
                        if len(en_data_wp_all)!= len(en_data_wp_clusterall_all):
                            en_data_wp.append(sum(en_data_wp_clusterall_all)/len(en_data_wp_clusterall_all))
                            zh_data_wp += zh_data_wp


                elif cluster_flag=='cluster_wa_sim':
                    zh_data_wp = [sum(wordpair2avgemb[wordpair]['zh2en']) / len(wordpair2avgemb[wordpair]['zh2en'])]
                    en_data_wp = [sum(wordpair2avgemb[wordpair]['en2zh']) / len(wordpair2avgemb[wordpair]['en2zh'])]
                    zh_data_wp=[cos_avg(zh_data_wp,wordpair2avgemb[wordpair[0]]['zh2en'])]
                    en_data_wp=[cos_avg(en_data_wp,wordpair2avgemb[wordpair[1]]['en2zh'])]
                elif cluster_flag=='cluster_wa_sim_mean':
                    zh_data_wp = [sum(wordpair2avgemb[wordpair]['zh2en']) / len(wordpair2avgemb[wordpair]['zh2en'])]
                    en_data_wp = [sum(wordpair2avgemb[wordpair]['en2zh']) / len(wordpair2avgemb[wordpair]['en2zh'])]
                    zh_wa_means=extract_wa_means(wordpair2avgemb,wordpair[0],'zh2en')
                    en_wa_means=extract_wa_means(wordpair2avgemb,wordpair[1],'en2zh')
                    zh_data_wp = [cos_avg(zh_data_wp, zh_wa_means)]
                    en_data_wp = [cos_avg(en_data_wp, en_wa_means)]

                elif cluster_flag=='cluster_wa_type':
                    zh_data_wp=[sum(wordpair2avgemb[wordpair[0]]['zh2en'])/len(wordpair2avgemb[wordpair[0]]['zh2en'])]
                    en_data_wp = [sum(wordpair2avgemb[wordpair]['en2zh']) / len(wordpair2avgemb[wordpair]['en2zh'])]

                elif cluster_flag=='token_type':
                    en_data_wp = wordpair2avgemb[wordpair]['en2zh']
                    zh_data_wp = [sum(wordpair2avgemb[wordpair[0]]['zh2en']) / len(wordpair2avgemb[wordpair[0]]['zh2en'])]*len(en_data_wp)
                elif cluster_flag=='token':
                    zh_data_wp = wordpair2avgemb[wordpair]['zh2en']
                    en_data_wp = wordpair2avgemb[wordpair]['en2zh']
                elif cluster_flag=='token_random':
                    zh_data_wp=deepcopy(wordpair2avgemb[wordpair]['zh2en'])
                    random.shuffle(zh_data_wp)
                    en_data_wp = wordpair2avgemb[wordpair]['en2zh']
                # cluster sim
                elif cluster_flag=='cluster_sim':
                    if src=='en':
                        transformed_en=trainer.apply_init_src_model(wordpair2avgemb[wordpair[1]]['en2zh'])
                        pass

                # cluster wa sample
                # zh_data_wp = []
                # en_data_wp = []
                # sample_size = math.ceil(wordpair2avgemb[wordpair]['count'] / 2)
                # for i in range(sample_times):
                #     random.seed(i)
                #     zh_data_wp.append(sum(random.sample(wordpair2avgemb[wordpair]['zh2en'], sample_size)) / sample_size)
                #     random.seed(i)
                #     en_data_wp.append(sum(random.sample(wordpair2avgemb[wordpair]['en2zh'], sample_size)) / sample_size)

                # cluster data distr
                # zh_data_wp=[sum(wordpair2avgemb[wordpair]['zh2en'])/len(wordpair2avgemb[wordpair]['zh2en'])]*len(wordpair2avgemb[wordpair]['zh2en'])
                # en_data_wp=[sum(wordpair2avgemb[wordpair]['en2zh'])/len(wordpair2avgemb[wordpair]['en2zh'])]*len(wordpair2avgemb[wordpair]['en2zh'])
                # print (wordpair,len(wordpair2avgemb[wordpair]['zh2en']),len(wordpair2avgemb[wordpair]['en2zh']))

                # clusterall one2one

                elif cluster_flag=='unsup_cluster':
                    if wordpair[1]+'|'+str(0) not in wordpair2avgemb:
                        zh_data_wp,en_data_wp= clusterall(mean_size,wordpair2avgemb,wordpair,zh_data_labels,en_data_labels)
                    else:
                        zh_data_wp=[]
                        en_data_wp=[]
                        for i in range(num_cluster):
                            en_data_wp_all = wordpair2avgemb.get(wordpair[1]+'|'+str(i)).get('en2zh')
                            zh_cluster=wordpair2avgemb.get(wordpair[1]+'|'+str(i)).get('2zhcluster')
                           

                            zh_data_wp_all = wordpair2avgemb.get(wordpair[0]+'|'+str(zh_cluster)).get('zh2en')
                            zh_data_wp.append(sum(zh_data_wp_all)/len(zh_data_wp_all))
                            en_data_wp.append(sum(en_data_wp_all)/len(en_data_wp_all))

                elif cluster_flag=='clusterall_minibatch':
                    num = mean_size
                    zh_data_wp_all = wordpair2avgemb[wordpair[0]]['zh2en']
                    en_data_wp_all = wordpair2avgemb[wordpair[1]]['en2zh']
                    if len(zh_data_wp_all)>5:
                        batches=int(len(zh_data_wp_all)/5)
                        if batches>5:
                            batches=5
                        batchsize=int(len(zh_data_wp_all)/batches)
                        start=0
                        for _ in range(batches):
                            end=start+batchsize
                            zh_data_wp.append(sum(zh_data_wp_all[start:end])/batchsize)
                            en_data_wp.append(sum(en_data_wp_all[start:end])/batchsize)
                            start=end
                    else:
                         zh_data_wp=[sum(zh_data_wp_all)/len(zh_data_wp_all)]
                         en_data_wp=[sum(en_data_wp_all)/len(en_data_wp_all)]
                   

                    


                   
                elif cluster_flag=='type':

                    num = mean_size
                    zh_data_wp_all = wordpair2avgemb[wordpair[0]]['zh2en']
                    en_data_wp_all = wordpair2avgemb[wordpair[1]]['en2zh']

                    if len(zh_data_wp_all) > num:
                        random.seed(0)
                        zh_data_wp_all=random.sample(zh_data_wp_all,num)
                    if len(en_data_wp_all) > num:
                        random.seed(0)
                        en_data_wp_all = random.sample(en_data_wp_all, num)

                    zh_data_wp=[sum(zh_data_wp_all)/len(zh_data_wp_all)]
                    en_data_wp=[sum(en_data_wp_all)/len(en_data_wp_all)]
                    zh_data_labels.append(wordpair[0])
                    en_data_labels.append(wordpair[1])
                elif cluster_flag=='clusterall_distr':
                    zh_data_wp_all = wordpair2avgemb[wordpair[0]]['zh2en']
                    en_data_wp_all = wordpair2avgemb[wordpair[1]]['en2zh']
                    zh_data_wp = [sum(zh_data_wp_all) / len(zh_data_wp_all)]*len(wordpair2avgemb[wordpair]['zh2en'])
                    en_data_wp = [sum(en_data_wp_all) / len(en_data_wp_all)]*len(wordpair2avgemb[wordpair]['en2zh'])
                #random cluster sample
                # zh_data_wp=[]
                # en_data_wp=[]
                # sample_size=math.ceil(wordpair2avgemb[wordpair]['count']/2)
                # for i in range(sample_times):
                #     zh_data_wp.append(sum(random.sample(wordpair2avgemb[wordpair[0]]['zh2en'], sample_size))/sample_size)
                #     en_data_wp.append(sum(random.sample(wordpair2avgemb[wordpair[1]]['en2zh'], sample_size))/sample_size)

                #random cluster one2one (wa len)
                elif cluster_flag=='cluster_wa_random':
                     zh_data_wp_len=len(wordpair2avgemb[wordpair]['zh2en'])
                     en_data_wp_len=len(wordpair2avgemb[wordpair]['en2zh'])
                     zh_data_wp=[sum(random.sample(wordpair2avgemb[wordpair[0]]['zh2en'], zh_data_wp_len))/zh_data_wp_len]
                     en_data_wp=[sum(random.sample(wordpair2avgemb[wordpair[1]]['en2zh'], en_data_wp_len))/en_data_wp_len]
                     add_labels(zh_data_wp_len, en_data_wp_len, zh_data_labels, en_data_labels, wordpair,
                               wordpair2avgemb)
                # cluster wsd one2one

                elif cluster_flag=='cluster_wsd':
                    zh_emb, en_emb, zh_wsd, en_wsd=cluster_wsd(en2zh_muse, zh2en_muse, wordpair, wordpair2avgemb)
                    # if len(en_emb)<5:
                    #     en_emb=wordpair2avgemb[wordpair[1]]['en2zh']
                    # if len(zh_emb)<5:
                    #     zh_emb=wordpair2avgemb[wordpair[0]]['zh2en']
                    # if zh_wsd=='wsd':
                    #     np.save('cluster_wsd_emb/{0}.{1}'.format(wordpair[0],wordpair[1]), zh_emb)
                    # if en_wsd=='wsd':
                    #     np.save('cluster_wsd_emb/{0}.{1}'.format(wordpair[1],wordpair[0]),en_emb)
                    zh_data_wp=[sum(zh_emb)/len(zh_emb)]
                    en_data_wp=[sum(en_emb)/len(en_emb)]
                    add_labels(len(zh_emb), len(en_emb), zh_data_labels, en_data_labels, wordpair,
                               wordpair2avgemb)

                elif cluster_flag=='cluster_wsd_random':
                    zh_emb, en_emb, zh_wsd, en_wsd = cluster_wsd(en2zh_muse, zh2en_muse, wordpair, wordpair2avgemb)
                    # if len(en_emb)<5:
                    #     en_emb=wordpair2avgemb[wordpair[1]]['en2zh']
                    # if len(zh_emb)<5:
                    #     zh_emb=wordpair2avgemb[wordpair[0]]['zh2en']
                    random.seed(0)
                    zh_emb=random.sample(wordpair2avgemb[wordpair[0]]['zh2en'],len(zh_emb))
                    random.seed(0)
                    en_emb=random.sample(wordpair2avgemb[wordpair[1]]['en2zh'],len(en_emb))
                    zh_data_wp = [sum(zh_emb) / len(zh_emb)]
                    en_data_wp = [sum(en_emb) / len(en_emb)]
                    add_labels(len(zh_emb), len(en_emb), zh_data_labels, en_data_labels, wordpair,
                           wordpair2avgemb)
                #random cluster wsd (wa len) sample
                # zh_data_wp=[]
                # en_data_wp=[]
                # zh_emb, en_emb=cluster_wsd(en2zh_muse, zh2en_muse, wordpair, wordpair2avgemb)
                #
                # sample_size=math.ceil(wordpair2avgemb[wordpair]['count']/2)
                # for i in range(sample_times):
                #     zh_data_wp.append(sum(random.sample(zh_emb, sample_size))/sample_size)
                #     en_data_wp.append(sum(random.sample(en_emb, sample_size))/sample_size)

                # random cluster wsd one2one (wa len)
                # zh_emb, en_emb=cluster_wsd(en2zh_muse, zh2en_muse, wordpair, wordpair2avgemb)
                # sample_size=wordpair2avgemb[wordpair]['count']
                #zh_data_wp=[sum(random.sample(zh_emb, sample_size))/sample_size)]
                #en_data_wp=[(sum(random.sample(en_emb, sample_size))/sample_size)]

                # print (wordpair,wordpair2avgemb[wordpair]['count'])

                zh_data+=zh_data_wp
                en_data+=en_data_wp
            else:
                print ('filtered:', wordpair)
    print (len(zh_data),len(en_data))
    return zh_data,en_data,zh_data_labels,en_data_labels
def create_wordpair2avgemb(wordpair2avgemb,zh_w,en_w,zh_w_data,en_w_data):
    # zh_w=zh_line.split()[int(zh_index)].lower()
    # en_w=en_line.split()[int(en_index)].lower()
    if type(zh_w_data)!=type(None) and type(en_w_data)!=type(None):
        wordpair2avgemb[(zh_w,en_w)]['zh2en'].append(zh_w_data)
        wordpair2avgemb[(zh_w,en_w)]['en2zh'].append(en_w_data)
        if 'count' in wordpair2avgemb[(zh_w,en_w)]:
            wordpair2avgemb[(zh_w,en_w)]['count']+=1
        else:
            wordpair2avgemb[(zh_w,en_w)]['count']=1

        wordpair2avgemb[zh_w]['zh2en'].append(zh_w_data)
        wordpair2avgemb[zh_w]['zh2en-transgold'].append(en_w)
        wordpair2avgemb[en_w]['en2zh'].append(en_w_data)
        wordpair2avgemb[en_w]['en2zh-transgold'].append(zh_w)




def create_muse_dict_filter(muse_dict):
    en2zh=defaultdict(lambda: defaultdict(list))
    zh2en=defaultdict(lambda: defaultdict(list))
    with open(muse_dict) as f:
        for line in f:
            en,zh,en_wsd,zh_wsd=line.split('\t')
            en, zh, en_wsd, zh_wsd = en.strip(), zh.strip(), en_wsd.strip(), zh_wsd.strip()
            en2zh[en]['zh'].append(zh)
            if en_wsd:
                en2zh[en]['wsd'].append(int(en_wsd))
            zh2en[zh]['en'].append(en)
            if zh_wsd:
                zh2en[zh]['wsd'].append(int(zh_wsd))
    return en2zh, zh2en


def add_lemmavariant_dict(en2zh_muse,zh2en_muse,zh_w,en_w):
    if en_w in en2zh_muse and zh_w in en2zh_muse.get(en_w).get('zh'):
        pass
    else:
        en_w_lemma = lemmatize(en_w, 'en')
        zh_w_lemma = lemmatize(zh_w, args.lg)

        if en_w_lemma in en2zh_muse and zh_w_lemma in en2zh_muse.get(en_w_lemma).get('zh'):
            if en_w not in zh2en_muse.get(zh_w_lemma).get('en'):
                zh2en_muse[zh_w_lemma]['en'].append(en_w)
                if zh2en_muse.get(zh_w_lemma).get('wsd'):
                    zh2en_muse[zh_w_lemma]['wsd'].append(max(zh2en_muse.get(zh_w_lemma).get('wsd'))+1)
            if zh_w not in en2zh_muse.get(en_w_lemma).get('zh'):
                en2zh_muse[en_w_lemma]['zh'].append(zh_w)
                if en2zh_muse.get(en_w_lemma).get('wsd'):
                    en2zh_muse[en_w_lemma]['wsd'].append(max(en2zh_muse.get(en_w_lemma).get('wsd'))+1)

            zh2en_muse[zh_w]=zh2en_muse.get(zh_w_lemma)
            en2zh_muse[en_w]=en2zh_muse.get(en_w_lemma)

def dict_filter(args,en2zh_muse,zh2en_muse,zh_data_dict,en_data_dict,zh_w,en_w):

    #filter zh_data_vocab and en_data_vocab
    if zh_data_dict and en_data_dict:
        if zh_w not in zh_data_dict or en_w not in en_data_dict:
            return False

    if args.muse_dict_filter:
        add_lemmavariant_dict(en2zh_muse,zh2en_muse,zh_w,en_w)
        en_data_dict = en2zh_muse
        if en_w in en_data_dict:
            zh_data_dict = en2zh_muse.get(en_w).get('zh')
        else:
            zh_data_dict = {}


    if type(zh_data_dict)==type(None) and type(en_data_dict) ==type(None) and not args.muse_dict_filter:
        return True # no filter
    elif zh_w in zh_data_dict and en_w in en_data_dict:
        return True
    else:
        return False

def fiter_typedict_with_vocabfile(zh_vocab_f,zh_data_dict):
    zh_data_dict_vocabfile = construct_types_dict_from_vocabfile(zh_vocab_f)
    for zh_w in list(zh_data_dict.keys()):
        if zh_w not in zh_data_dict_vocabfile:
            del zh_data_dict[zh_w]

def extract_common_text(args,src_data_f_lsts,tgt_data_f_lst):
    para=open(args.para).readlines()
    align=open(args.align).readlines()
    para_out=[]
    align_out=[]
    for i,line in enumerate(para):
            start = timer()
            # ...
            print ('.', end='',flush=True)
            src_line,tgt_line=line.strip().split(' ||| ')
            skip_line=False
            for src_data_f_lst in src_data_f_lsts:
                src_line_data_all=extract_emb(src_data_f_lst,src_line)
                tgt_line_data_all=extract_emb(tgt_data_f_lst,tgt_line)
                if type(src_line_data_all)==type(None) or type(tgt_line_data_all)==type(None):
                    print ('line is empty',src_line,tgt_line)
                    skip_line=True
                    break
                if len(src_line_data_all) != len(src_line.split()) or len(tgt_line_data_all)!=len(tgt_line.split()):
                    skip_line=True
                    break
            if not skip_line:
                para_out.append(line)
                align_out.append(align[i])
                if len(para_out)>args.batch_store:
                    return para_out,align_out
    return para_out,align_out

def produce_crossling_batch(args,para,align,zh_data_f_lst,en_data_f_lst,batchsize=None):
        # muse dict filter:
        if args.muse_dict_filter:
            en2zh_muse,zh2en_muse=create_muse_dict_filter(args.muse_dict_filter)
        else:
            en2zh_muse={}
            zh2en_muse={}
        if batchsize==None:
            batchsize=args.batch_store
        # para, align, zh_data_lst, en_data_lst=args.para, args.align, args.tgt_data, args.src_data
        # print ('zh data: {0}'.format(repr(zh_data_lst)))
        # print ('en data" {0}'.format(repr(en_data_lst)))

        zh_data=[]
        en_data=[]
        wordpair2avgemb = defaultdict(lambda: defaultdict(list))  # cluster contexts according to translation
        # align_f_gen=file2generator(align)
        line_counter=0
        # key_exist_ch=None
        # key_exist_en=None
        if args.type==True:
            zh_data_dict=construct_types_dict(zh_data_f_lst)
            en_data_dict=construct_types_dict(en_data_f_lst)
            if args.src_vocab and args.tgt_vocab:
                fiter_typedict_with_vocabfile(args.src_vocab, zh_data_dict)
                fiter_typedict_with_vocabfile(args.tgt_vocab, en_data_dict)


        elif args.src_vocab and args.tgt_vocab:
            zh_data_dict=construct_types_dict_from_vocabfile(args.src_vocab)
            en_data_dict=construct_types_dict_from_vocabfile(args.tgt_vocab)
        else:
            zh_data_dict=None
            en_data_dict=None
        for i,line in enumerate(para):
            start = timer()
            # ...
            print ('.', end='',flush=True)
            zh_line,en_line=line.strip().split(' ||| ')

            align_line=align[i]

            if args.type==True:
                zh_line_data, en_line_data,zh_line_data_all,en_line_data_all=retrieve_type_emb(zh_line,en_line,align_line,zh_data_dict,en_data_dict,wordpair2avgemb,en2zh_muse)
            elif args.sent_emb==True:

                zh_line_data,en_line_data=retrieve_sentemb(zh_line,en_line,zh_data_f_lst,en_data_f_lst)
            else: #token-level alignment
                zh_line_data=[]
                en_line_data=[]
                zh_line_data_all=extract_emb(zh_data_f_lst,zh_line)
                en_line_data_all=extract_emb(en_data_f_lst,en_line)
                # if type(zh_line_data_all)==type(None) or type(en_line_data_all)==type(None):
                #     print ('line is empty',zh_line,en_line)
                #     continue
                # if len(zh_line_data_all) != len(zh_line.split()) or len(en_line_data_all)!=len(en_line.split()):
                #     continue
                for pair in align_line.strip().split():
                    zh_index, en_index = pair.split('-')
                    zh_w=zh_line.split()[int(zh_index)]
                    en_w=en_line.split()[int(en_index)]
                    # if args.muse_dict_filter:
                    #     en_data_dict=en2zh_muse
                    #     if en_w in en_data_dict:
                    #         zh_data_dict=en2zh_muse[en_w]['zh']
                    #     else:
                    #         zh_data_dict={}
                    # if zh_w in zh_data_dict and en_w in en_data_dict:
                    if dict_filter(args,en2zh_muse,zh2en_muse,zh_data_dict,en_data_dict,zh_w,en_w):
                        zh_line_data.append(zh_line_data_all[int(zh_index)])
                        en_line_data.append(en_line_data_all[int(en_index)])
                        create_wordpair2avgemb(wordpair2avgemb,zh_w,en_w,zh_line_data_all[int(zh_index)],en_line_data_all[int(en_index)])
            if args.sent_avg==True:
                zh_line_data=[sum(zh_line_data_all)/len(zh_line_data_all)]
                en_line_data=[sum(en_line_data_all)/len(en_line_data_all)]
            if not zh_line_data or not en_line_data:
                continue
            zh_data+=zh_line_data
            en_data+=en_line_data



            line_counter+=1
            if line_counter%10000==0 and line_counter>10000:
                print (line_counter)
            if line_counter>=batchsize:
                print ('yielding batch')
                # if args.cluster:
                #     zh_data_init,en_data_init=produce_avgemb_cluster(wordpair2avgemb,en2zh_muse,zh2en_muse,args.cluster[0],zh_data,en_data)
                #
                #     # zh_data_second,en_data_second=produce_avgemb_cluster(wordpair2avgemb,en2zh_muse,zh2en_muse,args.cluster[1],zh_data,en_data)
                # else:
                #     zh_data_init,en_data_init=zh_data,en_data
                #     zh_data_second,en_data_second=None,None

                yield zh_data,en_data,wordpair2avgemb,zh2en_muse,en2zh_muse
                zh_data=[]
                en_data=[]
            end = timer()
            # print ('line processed:',end-start)


        if zh_data!=[]:
            # if args.cluster:
            #     zh_data_init, en_data_init = produce_avgemb_cluster(wordpair2avgemb, en2zh_muse, zh2en_muse,
            #                                                         args.cluster[0])
            #     zh_data_second, en_data_second = produce_avgemb_cluster(wordpair2avgemb, en2zh_muse, zh2en_muse,
            #                                                                 args.cluster[1],zh_data,en_data)
            # else:
            #     zh_data_init, en_data_init = zh_data, en_data
            #     zh_data_second, en_data_second = None, None

            yield zh_data, en_data, wordpair2avgemb,zh2en_muse,en2zh_muse
        # close_h5pyfile_lst(zh_data_f_lst)
        # close_h5pyfile_lst(en_data_f_lst)










def produce_batch(data_src,data_tgt,batch):
    for i in range(int(len(data_src)/batch)+1):
        if i<len(data_src):
            if i+batch>len(data_src):
                yield data_src[i:], data_tgt[i:]
            else:
                yield data_src[i:i+batch], data_tgt[i:i+batch]

def eval_and_save(trainer,best,i,args):
    eval('scws',args,trainer)
    eval('usim',args,trainer)
    eval_wic(args,trainer)
    # eval_wic(args,trainer,crossling_flag=True)
    # eval_wic(args,trainer,crossling_flag=True,align_flag=True)
    eval_context_simlex(args,trainer)
    # eval_lexsub(trainer,args,LEXSUB_EN,'')

    '''
    if args.lg in ['zh','es']:
        trainer.eval_bdi(args,BDI_MUSE)
        if trainer.base_embed=='fasttext':
            simlex_rho = trainer.eval('simlex', args)
            trainer.eval_bdi(args, 'sentence_type')
            trainer.eval_bdi(args, P_BDI)
            trainer.eval_bdi(args, NP_BDI)
            # scws_vocab_rho = trainer.eval('scws.vocab', args)
            bcws_vocab_rho = trainer.eval('bcws.vocab', args)
        else:
            simlex_rho = trainer.eval('simlex', args)
            # scws_rho = trainer.eval('scws', args)
            # scws_vocab_rho=trainer.eval('scws.vocab',args)
            bcws_rho = trainer.eval('bcws', args)
            bcws_vocab_rho=trainer.eval('bcws.vocab',args)
            usim_rho=trainer.eval('usim',args)


            if args.sent_emb:
                trainer.eval_bdi(args,'sentence_emb')
            else:
                trainer.eval_bdi(args,'sentence')

            
            if args.type or trainer.cluster=='clusterall':
                trainer.eval_bdi(args,'sentence_type')
                trainer.eval_bdi(args, P_BDI)
                trainer.eval_bdi(args, NP_BDI)
    
                if args.avg_flag:
                    # trainer.eval_bdi(args, P_CBDI_TYPE_CONTEXT_AVG)
                    trainer.eval_bdi(args, P_CBDI_TYPE_CONTEXT_TGT_AVG)
                    # trainer.eval_bdi(args, NP_CBDI_TYPE_CONTEXT_AVG)
                    trainer.eval_bdi(args, NP_CBDI_TYPE_CONTEXT_TGT_AVG)
            # else:
            #     trainer.eval_bdi(args,P_CBDI_TOK_CONTEXT_AVG)
            #     trainer.eval_bdi(args,NP_CBDI_TOK_CONTEXT_AVG)
            
            trainer.eval_bdi(args, P_CBDI)
            
            trainer.eval_bdi(args, NP_CBDI)
            trainer.eval_bdi(args,NP_CLS)
            trainer.eval_bdi(args,NP_CLS_type)
            trainer.eval_lexsub(args,LEXSUB_EN,'')
            trainer.eval_lexsub(args,LEXSUB_EN,'orig')


            if args.lg=='es':
                trainer.eval_lexsub(args, LEXSUB_CL,'')

    elif args.lg=='de':
        # trainer.eval_bdi(args,BDI_MUSE_DE)
        
        for bdi_longtail in BDI_LONGTAILS_all:
            trainer.eval_bdi(args, bdi_longtail, 'orig')
            trainer.eval_bdi(args,bdi_longtail)

        if args.avg_flag:
            for bdi_longtail in BDI_LONGTAILS_1+BDI_LONGTAILS_5:
                trainer.eval_bdi(args, bdi_longtail, 'orig')
                trainer.eval_bdi(args, bdi_longtail)
        
        trainer.eval_bdi_mt(args,BDI_LONGTAIL_MT)

        # trainer.eval_bdi(args,BDI_LONGTAIL_GENERAL+'_all')
        # trainer.eval_bdi(args,BDI_LONGTAIL_GENERAL,'orig')
        #
        # trainer.eval_bdi(args,BDI_LONGTAIL_NEWS+'_all')
        # trainer.eval_bdi(args,BDI_LONGTAIL_NEWS,'orig')
        #
        # trainer.eval_bdi(args,BDI_LONGTAIL_HIML+)
        # trainer.eval_bdi(args,BDI_LONGTAIL_HIML,'orig')
        #
        # trainer.eval_bdi(args,BDI_LONGTAIL_RAREHIML)
        # trainer.eval_bdi(args,BDI_LONGTAIL_RAREHIML,'orig')
        '''









        # trainer.eval_bdi(args,NP_CLS_WSD)
        # trainer.eval_bdi(args,NP_CLS_WSD_type)
        # if args.batch_store>=100000:
        #     trainer.eval_bdi(args,P_CBDI_200K)
        #     trainer.eval_bdi(args, NP_CBDI_200K)

        # trainer.eval_bdi(args, NP_BDI_SOURCE)

    # trainer.eval_bdi(args,P_CBDI_TOK_AVG)





    # trainer.eval_bdi(args,'bdi_sense_typeavg')



    #
    if args.save == True:
        # trainer.save(scws_rho, best, 'scws', i)
        trainer.save(bcws_rho, best, 'bcws', i)


def process_input_data(trainer,data_src_store,data_tgt_store,args):
    # if trainer.src == 'en':
    #     data_src_store = en_data
    #     data_tgt_store = zh_data
    # elif trainer.src in ['zh', 'es']:
    #     data_src_store = zh_data
    #     data_tgt_store = en_data
    if args.src_mean :
        trainer.src_mean=torch.from_numpy(np.load(args.src_mean)).type('torch.FloatTensor').to(trainer.device)
    if args.tgt_mean:
        trainer.tgt_mean=torch.from_numpy(np.load(args.tgt_mean)).type('torch.FloatTensor').to(trainer.device)
    data_src_store = torch.from_numpy(np.vstack(data_src_store)).type('torch.FloatTensor').to(trainer.device)
    src_mean = normalize_embeddings(data_src_store, args.norm, trainer.src_mean)
    trainer.src_mean = src_mean

    # src_mean = normalize_embeddings(data_src_store, args.norm)
    # tgt_mean = normalize_embeddings(data_tgt_store, args.norm)
    # trainer.src_mean = src_mean
    # trainer.tgt_mean = tgt_mean
    if type(data_tgt_store) !=type(None):
        data_tgt_store = torch.from_numpy(np.vstack(data_tgt_store)).type('torch.FloatTensor').to(trainer.device)
        tgt_mean = normalize_embeddings(data_tgt_store, args.norm, trainer.tgt_mean)
        trainer.tgt_mean = tgt_mean
        if type(tgt_mean)!=type(None):
            np.save(trainer.tag+'_mean_tgt.npy',np.array(tgt_mean.cpu()))
            np.save(trainer.tag+'_mean_src.npy',np.array(src_mean.cpu()))


    return data_src_store,data_tgt_store

def src_tgt_data(trainer,en_data,zh_data):
    if trainer.src == 'en':
        data_src_store = en_data
        data_tgt_store = zh_data
    elif trainer.tgt =='en':
        data_src_store = zh_data
        data_tgt_store = en_data
    return data_src_store, data_tgt_store

def store_training(zh_data,en_data,zh_data_labels,en_data_labels,tag,lg):
    if len(zh_data_labels)==0:
        pass
    else:
        assert len(zh_data)==len(zh_data_labels)==len(en_data)==len(en_data_labels)
        with open(os.path.join('./training_data/','{0}.{1}.vec'.format(tag,'en')),'w') as f_en, open(os.path.join('./training_data/','{0}.{1}.vec'.format(tag,lg)),'w') as f_zh:
            f_en.write('{0} {1}\n'.format(str(len(en_data_labels)), len(en_data[0])))
            f_zh.write('{0} {1}\n'.format(str(len(zh_data_labels)), len(zh_data[0])))

            for i,data in enumerate(zh_data):
                f_zh.write('{0}'.format(zh_data_labels[i]) + ' ' + ' '.join([str(v.item()) for v in data]) + '\n')
                f_en.write('{0}'.format(en_data_labels[i]) + ' ' + ' '.join([str(v.item()) for v in en_data[i]]) + '\n')



def train_init(trainer_init,trainer,X_src_init,X_tgt_init,args):
    trainer_init.model_type = args.init_model[0]
    if type(trainer_init.src_mean)==type(None):
            trainer_init.src_mean=trainer.src_mean
            trainer_init.tgt_mean=trainer.tgt_mean
    trainer.init_src_model = trainer_init.model_src
    trainer_init.model_update(X_src_init, X_tgt_init, -2)

def train_iter(args, tgt_data_orig,src_data_origs,wordpair2avgembs,en2zh_muse,zh2en_muse,trainer,batch_store_count,best,cluster_flag,freq_thres):
    trainers=[]
    X_src_inits=[]
    skip_eval=False

    for src_data_orig,wordpair2avgemb in zip(src_data_origs,wordpair2avgembs):
        
        if cluster_flag:
            src_data, tgt_data, src_data_labels, tgt_data_labels= produce_avgemb_cluster(args.meansize,args.cluster_wsd_thres,freq_thres,wordpair2avgemb, en2zh_muse, zh2en_muse,
                                                  cluster_flag, src_data_orig, tgt_data_orig, args.percent)
        else:
            tgt_data, src_data=tgt_data_orig,src_data_origs

        trainer = Trainer(D_in_src, D_in_tgt, D_out, args, device,trainer.tag,cluster_flag)
        X_src_init,X_tgt_init=process_input_data(trainer, src_data, tgt_data, args)
        # src_datas.append(src_data)
        if args.store_train:
            store_training(tgt_data, src_data, tgt_data_labels, src_data_labels, trainer.tag, args.lg)
        
        if args.init_model != '' and batch_store_count == 0:
            
            trainer_init = Trainer(D_in_src, D_in_tgt, D_out, args, device,trainer.tag,cluster_flag)
            print('training data mean cosin similarity',np.mean(np.array(produce_cosine_list(X_src_init[:100],X_tgt_init[:100]))))
            train_init(trainer_init,trainer,X_src_init,X_tgt_init,args)
            X_src_temp,X_tgt_temp=trainer_init.apply_model(X_src_init,X_tgt_init)
            print('training data mean cosin similarity',np.mean(np.array(produce_cosine_list(X_src_temp[:100],X_tgt_temp[:100]))))
            print('evaluation after init model')
            if not skip_eval:
                eval_and_save(trainer_init, best, -2, args)
            trainers.append(trainer)
            X_src_inits.append(X_src_init)
            skip_eval=True
                # if args.src_data2:
                #     trainer2 = Trainer(D_in_src, D_in_tgt, D_out, args, device,trainer.tag,cluster_flag)
                #     train_init(trainer_init2,trainer2,X_src2_init,X_tgt_init,args)
                #     print('evaluation after init model for src2')
                #     eval_and_save(trainer_init2, best, -2, args)

    if len(args.init_model) >= 2 and args.init_model[1] in [MIM, EXACT_ORTHO]: 
        
        X_src_transformeds=[]
        for trainer,X_src_init in zip(trainers,X_src_inits):
            trainer.model_type = args.init_model[1]
            X_src_transformed = trainer.apply_init_src_model(X_src_init)
            X_src_transformeds.append(X_src_transformed)
        X_src_transformed=sum(X_src_transformeds)/len(X_src_transformeds)
        trainer.model_update(X_src_transformed, X_tgt_init, -1)
        # trainer.model_type = args.model
        print('evaluation after adjusting cwe through mim')
        if args.error_output:
            args.error_output = args.error_output + '_mim'
        trainer.tag=trainer.tag+'_'+args.init_model[1]
        if args.tgt:
            torch.save(trainer.model_tgt.state_dict(), trainer.tag+'.pt')
        
        trainer.cluster=cluster_flag
        eval_and_save(trainer, best, -1, args)

def semisupervise_error(word,result,gold,trans_candi,f1_dict):
    per_trans_res=defaultdict(lambda: defaultdict(int))
    for i,label in enumerate(gold):
        pred=trans_candi[result[i]]
        if label.lower()==pred.lower():
            per_trans_res[label]['correct']+=1
        per_trans_res[label]['gold_count']+=1
        per_trans_res[pred]['pred_count']+=1
    for w in per_trans_res:
        f1_dict['correct'].append(per_trans_res[w]['correct'])
        if per_trans_res[w]['pred_count']==0:
            precision=0
        else:
            precision=per_trans_res[w]['correct']/float(per_trans_res[w]['pred_count'])
        recall=per_trans_res[w]['correct']/float(per_trans_res[w]['gold_count'])
        if precision==0 or recall==0:
            f1=0
        else:
            f1=2*(precision*recall)/(precision+recall)
        f1_dict['f1'].append(f1)
        f1_dict['word'].append((word,w))
        print (w, 'correct:', per_trans_res[w]['correct'], 'gold_count', per_trans_res[w]['gold_count'], 'pred count', per_trans_res[w]['pred_count'])



def update_wordpair2avgemb_semisupervise(trainer_first,en2zh_muse,zh2en_muse,wordpair2avgemb,src,freq_thres):
    # delete gold wa parallel

    lg2dict={'en2zh':en2zh_muse,'zh2en':zh2en_muse}
    for wordpair in list(wordpair2avgemb.keys()):
        if type(wordpair) == tuple:
            del wordpair2avgemb[wordpair]
    for word in list(wordpair2avgemb.keys()):
        if type(word)!=tuple:
            for lg_tag in wordpair2avgemb[word]:
                if lg_tag in ['zh2en','en2zh']:#it's a chinese word
                    en=lg_tag.split('2')[1]
                    if word in lg2dict[lg_tag] and en in lg2dict[lg_tag][word]:
                        trans_candidates=lg2dict[lg_tag].get(word).get(en)
                        lg_tag_trans = en+'2'+lg_tag.split('2')[0]
                    else:
                        continue
                else:
                    continue
                trans_candidates = [w for w in trans_candidates if wordpair2avgemb.get(w) and wordpair2avgemb.get(w).get(lg_tag_trans)]
                if not trans_candidates:
                    continue

                trans_candi_emb_orig = torch.tensor(
                    [sum(wordpair2avgemb.get(w).get(lg_tag_trans)) / len(wordpair2avgemb.get(w).get(lg_tag_trans)) for w in
                     trans_candidates])
                current_w_emb_orig = torch.tensor(wordpair2avgemb.get(word).get(lg_tag)).to(trainer_first.device)
                if src == lg_tag.split('2')[0]:
                    current_w_emb,trans_candi_emb=trainer_first.apply_model(current_w_emb_orig,trans_candi_emb_orig)
                elif src==lg_tag.split('2')[1]:
                    trans_candi_emb, current_w_emb=trainer_first.apply_model(trans_candi_emb_orig,current_w_emb_orig)
                print ('===error analysis for word==', word)

                cos_matrix=produce_cos_matrix(current_w_emb,trans_candi_emb)
                print ('cos matrix',cos_matrix)
                result=torch.argmax(cos_matrix,1)
                golds=wordpair2avgemb.get(word).get(lg_tag+'-transgold')
                golds_res = torch.tensor([trans_candidates.index(g) for g in golds])
                print ('gold res',golds_res)
                f1_dict=defaultdict(list)
                semisupervise_error(word,result,golds,trans_candidates,f1_dict)
                # wa_emb_weighted=trans_weighted_wa(cos_matrix, current_w_emb_orig,golds_res)


                for i,tran in enumerate(trans_candidates):

                    aligned_w = current_w_emb_orig[result == i]
                    aligned_w = aligned_w.detach().cpu().numpy()
                    if len(aligned_w)<freq_thres: # not enough data, then fall back to clusterall
                        aligned_w=['contextall']
                        # aligned_w=wa_emb_weighted[i]
                    if len(aligned_w)!=0:

                        if lg_tag=='zh2en':
                            wordpair2avgemb[(word,tran)][lg_tag]=list(aligned_w)
                        elif lg_tag=='en2zh':
                            wordpair2avgemb[(tran,word)][lg_tag]=list(aligned_w)

    print ('total correct', sum(f1_dict['correct']), 'average macro', sum(f1_dict['f1'])/len(f1_dict['f1']))


def trans_weighted_wa(cos_matrix,current_w_emb_orig,golds_res):
    wa_emb_weighted={}
    for i in range(cos_matrix.size()[1]):
        weights=cos_matrix[:,i]+1
        print (weights)
        sorted_sim = torch.argsort(-weights, dim=0)[:10]
        weights_current=weights[sorted_sim]
        current_w_emb_orig_current=current_w_emb_orig[sorted_sim,:]
        print ('sorted',weights_current)
        print ('gold annotate', i, golds_res[sorted_sim])
        wa_emb_weighted_current=sum(torch.transpose(torch.transpose(current_w_emb_orig_current,0,1)*weights_current,0,1))/sum(weights_current)
        wa_emb_weighted_current=wa_emb_weighted_current.resize(1,len(wa_emb_weighted_current))
        wa_emb_weighted[i]=wa_emb_weighted_current
    return wa_emb_weighted


def train(D_in_src,D_in_tgt,D_out,args,device):
    # src_data2,tgt_data2,wordpair2avgemb2=None,None,None
    tgt_data_f_lst = h5pyfile_lst(args.tgt_data)
    src_data_f_lsts=[]
    for src_data_f in args.src_data:
        src_data_f_lst = h5pyfile_lst([src_data_f])
        src_data_f_lsts.append(src_data_f_lst)
  

    # trainer.data_mean(args)
    best={'scws':0,'bcws':0,'scws_epoch':0,'bcws_epoch':0}

    batch_store_count = 0

    if args.cluster:
        align_level = args.cluster
    else:
        if args.type:
            align_level = ['type']
        else:
            align_level = ['token']
    para,align=extract_common_text(args,src_data_f_lsts,tgt_data_f_lst)
    src_datas=[]
    wordpair2avgembs=[]
    for src_data_f_lst in src_data_f_lsts:
        src_data,tgt_data,wordpair2avgemb,tgt2src_muse,src2tgt_muse=next(produce_crossling_batch(args,para,align,src_data_f_lst,tgt_data_f_lst))
        src_datas.append(src_data)
        wordpair2avgembs.append(wordpair2avgemb)
    # if args.src_data2:
    #     src_data_f_lst2=h5pyfile_lst(args.src_data2)
    #     tgt_src_data2=produce_crossling_batch(args,tgt_data_f_lst,src_data_f_lst2)
    # for tgt_src_data in tgt_src_datas:
        
    #  for tgt_data,src_data,wordpair2avgemb,en2zh_muse,zh2en_muse in tgt_src_data:
    # print ('processed {0} data'.format(batch_store_count*args.batch_store))
    print ('processed current batch with {0} data'.format(len(tgt_data)))
    print('=====initial training=====')

    tag="{0}_{1}_{2}_{3}_{4}".format(args.lg,args.base_embed+'~'+args.aligned_embed,args.batch_store,'_'.join(align_level)+'0',os.path.basename(args.muse_dict_filter))
    if args.cluster:
        cluster_flag=args.cluster[0]
    else:
        cluster_flag=None
    trainer_first = Trainer(D_in_src, D_in_tgt, D_out, args, device,tag,cluster_flag)

    train_iter(args, tgt_data, src_datas,wordpair2avgembs, tgt2src_muse, src2tgt_muse,trainer_first, batch_store_count, best,cluster_flag,args.freq_thres)
    # if args.cluster and len(args.cluster) > 1:
    #     print('======semi-supervised word alignment')
    #     tag = "{0}_{1}_{2}_{3}_{4}".format(args.lg,args.base_embed+'~'+args.tgt_base_embed, args.batch_store, '_'.join(align_level) + '1',
    #                                     os.path.basename(args.muse_dict_filter))

    #     freq_thres_second=args.freq_thres
    #     update_wordpair2avgemb_semisupervise(trainer_first,en2zh_muse,zh2en_muse,wordpair2avgemb,args.src,freq_thres_second)
    #     trainer_semisupervised = Trainer(D_in_src, D_in_tgt, D_out, args, device,tag,args.cluster[1])
    #     train_iter(args, tgt_data, src_data,wordpair2avgemb, en2zh_muse, zh2en_muse, trainer_semisupervised,
    #                 batch_store_count, best, args.cluster[1],freq_thres_second)


    # batch_store_count+=1

    # break

    close_h5pyfile_lst(tgt_data_f_lst)
    close_h5pyfile_lst(src_data_f_lst)


if __name__=='__main__':
    import argparse
    import os
    from sklearn.metrics.pairwise import cosine_similarity
    from math import inf
    from copy import deepcopy
    import yaml
    from sklearn.metrics import  silhouette_score

    import ast
    from collections import defaultdict

    from timeit import default_timer as timer
    import random
    from trainer import *
    from evaluation_script import *



    from variable_names import *
    from sklearn.cluster import KMeans
    
    
    args = argparse.ArgumentParser('crossling mapping')
    args.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    args.add_argument('--align',type=str, help='alignment files produced by fast align')
    args.add_argument('--para', type=str, help='training parallel data for word alignment')
    args.add_argument('--batch_store', default=200000,type=int, help='training batch size. Default is the whole training data size')

    args.add_argument('--tgt_data', type=str, nargs='+',help='tgtdata in npy')
    args.add_argument('--src_data', type=str, nargs='+',help='src 1 data in npy')
    # args.add_argument('--src_data2',type=str,nargs='+',help='src 2 data in npy')
    # args.add_argument('--base_embed',type=str,help='base embed type: elmo or bert')
    # args.add_argument('--test_zh_data',type =str, help='test chinese data')
    args.add_argument('--test_data_dir', default='./corpora/',type=str, help='test data dir')
    args.add_argument('--epoch', default=0, type=int, help='number of epochs')
    args.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    args.add_argument('--norm', default='', type=str, help='normalize mode')
    args.add_argument('--src_path', type=str, default=None,help='model path')
    args.add_argument('--save', type=ast.literal_eval, default=False, help='save flag for model path: True or False')
    # args.add_argument('--evaluation',default='',type=str, help='evaluation set')
    args.add_argument('--model', type=str, default=BIDIRECT_MODEL, help='model type: bidirection, unidirection, muse')
    args.add_argument('--src_lg', type=str,help='source language')
    args.add_argument('--init_model', default=[EXACT_ORTHO, MIM],type=str, help='init model', nargs='+')
    args.add_argument('--reg', type=float,default=0.0, help='regularization ratio')
    args.add_argument('--sent_avg', action='store_true', help='sentence embeddings from word average')
    args.add_argument('--sent_emb', action='store_true',help='sentence embeddings from bert')
    args.add_argument('--type', action='store_true', help='whether to use type level word as baselines')
    # args.add_argument('--lg', type=str,help='the other language than english: zh or es')
    # args.add_argument('--avg_flag',type=bool,default=False,help='if the input embeddings are average anchors')
    args.add_argument('--src_vocab', default=None,type=str, help='src vocabulary file to filter for training' )
    args.add_argument('--tgt_vocab', default=None,type=str, help='tgt vocabulary file to filter for training')
    args.add_argument('--muse_dict_filter', default='',type=str, help='muse dictionary filter. Only used in crosslingual alignment')
    args.add_argument('--cluster', type=str,default=['type'],nargs='+', help='whether to cluster token level embeddings. Default is type level alignment')
    args.add_argument('--error_output', type=str, default=None,help='Error analysis file output')
    args.add_argument('--percent', type=float, default=None,help='For aligning on sense embeddings. wsd percentage in the data')
    args.add_argument('--freq_thres', type=int, default=5, help='freq threshold for clustering sense level embedding')
    args.add_argument('--meansize', type=int, default=inf, help='the size of word representation for taking the average ')
    args.add_argument('--store_train',action='store_true', help='whether to store the training data for muse alignmnet')
    args.add_argument('--cluster_wsd_thres',default=None,help='For clustring sense level embeddings. ')
    # args.add_argument('--tgt_base_embed',type=str,help='target embedding model type ')
    args.add_argument('--src_mean',type=str,help='mean file to be loaded ')
    args.add_argument('--tgt_mean',type=str,help='mean file to be loaded ')
    args.add_argument('--baseline',type=str,default=None,help='whether to test concatenate baseline ')

    args.add_argument('--tgt_lg',type=str,help='tgt language')
    args.add_argument('--tgt',default=False,action='store_true', help='whether to use target side for evaluation')
    args.add_argument('--src',action='store_true', default=False,help='whether to use target side for evaluation')
    args.add_argument('--dim', type=int,help='dimension size of the base embed')
    args.add_argument('--base_embed_force', type=str, help='base embed model. See variable_names.py for details')
    args.add_argument('--eval',action='store_true',default=False,help='whether to perform evaluation')

    args=args.parse_args()

    # type level params
    args.context_num = ''
    args.avg_flag=False
    if os.path.basename(args.src_data[0]).startswith('average') and args.type:
        args.avg_flag = True
        if os.path.basename(args.src_data[0]).split('_')[1].isdigit():
            args.context_num=os.path.basename(args.src_data[0]).split('_')[1]+'_'

    # base embed name
    args.src_base_embed= ['.'.join(os.path.basename(src_data_f).split('.')[-4:-3]) for src_data_f in args.src_data ]
    args.tgt_base_embed='.'.join(os.path.basename(args.tgt_data[0]).split('.')[-4:-3])
    # if len(args.src_data)>1:
    #     args.src_base_embed2='.'.join(os.path.basename(args.src_data2[0]).split('.')[-4:-3])

    if args.src and not args.tgt:
        args.base_embed=args.src_base_embed[0]
        args.aligned_embed=args.tgt_base_embed
        args.lg=args.src_lg
    elif args.tgt and not args.src:
        args.base_embed=args.tgt_base_embed
        args.aligned_embed='+'.join(args.src_base_embed)
        args.lg=args.tgt_lg
       
    ## make sure that the input files follow src2tgt format
    assert args.para.split('.')[-1].split('2')[0]==args.src_lg and args.para.split('.')[-1].split('2')[1]==args.tgt_lg
    assert args.align.split('.')[-1].split('2')[0]==args.src_lg and args.para.split('.')[-1].split('2')[1]==args.tgt_lg
    if args.muse_dict_filter:
        assert args.muse_dict_filter('.')[-1].split('2')[0]==args.src_lg and args.para.split('.')[-1].split('2')[1]==args.tgt_lg
    # train
    if not args.dim:
        if args.base_embed.startswith('bert'):
            D_in_src=768
            D_in_tgt=768
            D_out=768
        elif args.base_embed=='elmo':
            D_in_src = 1024
            D_in_tgt = 1024
            D_out = 1024
        elif args.base_embed.startswith('fasttext'):
            D_in_src = 300
            D_in_tgt = 300
            D_out = 300

        if 'large' in args.base_embed:
            D_in_src = 1024
            D_in_tgt = 1024
            D_out = 1024
    else:
        D_in_src=args.dim
        D_in_tgt=args.dim
        D_out=args.dim

    device = torch.device("cuda:{0}".format(args.gpu) if torch.cuda.is_available() and args.gpu>-1 else "cpu")
    print(device)

    print ('init:',args.init_model, 'model:',args.model, args.norm,args.src_lg, 'gpu:',device)

    if args.base_embed_force:
        args.base_embed=args.base_embed_force

    train(D_in_src,D_in_tgt,D_out,args,device)

    
    #
    # test
    # trainer_muse=Trainer(D_in_src, D_in_tgt, D_out, args)
    # trainer_muse.load_eval(args)
