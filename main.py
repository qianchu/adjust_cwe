
def train(src_data,tgt_data,device):
    src_data=src_data.to(device)
    tgt_data=tgt_data.to(device)

    #1. Orthogonal
    trainer_ortho=Trainer(src_data.size()[1], tgt_data.size()[1], tgt_data.size()[1], device)
    trainer_ortho.model_type='ortho'
    trainer_ortho.exact_model_ortho_update(src_data, tgt_data)
    src_embeds_transformed,tgt_embeds_transformed=trainer_ortho.apply_model(src_data, tgt_data)

    #2. Mim
    trainer_mim=Trainer(src_data.size()[1], tgt_data.size()[1], tgt_data.size()[1], device)
    trainer_mim.model_type='mim'
    trainer_mim.mim_model_update(src_embeds_transformed, tgt_embeds_transformed)
    src_embeds_transformed2,tgt_embeds_transformed2=trainer_mim.apply_model(src_embeds_transformed,tgt_embeds_transformed)
    
    return trainer_ortho,trainer_mim

def preprocess_data(datatext,data_embed_src,data_embed_tgt):
    outputdata_src=[]
    outputdata_tgt=[]
    data_embed_src=np.load(data_embed_src,allow_pickle=True)[()]
    data_embed_tgt=np.load(data_embed_tgt,allow_pickle=True)[()]
    for line in open(datatext).readlines():
        line_key=produce_key(line)
        if line_key in data_embed_src and line_key in data_embed_tgt:
            assert len(data_embed_src[line_key])==len(line.split())==len(data_embed_tgt[line_key])
            outputdata_src.append(data_embed_src)
            outputdata_tgt.append(data_embed_tgt)
    print ('processed {0} embeddings',len(outputdata_src))
    return torch.from_numpy(np.concatenate(outputdata_src,0)),torch.from_numpy(np.concatenate(outputdata_tgt,0))

def preprocess_data_type(datatext,data_embed_src,data_embed_tgt,max_example=100):
    outputdata_src=[]
    outputdata_tgt=[]
    w2type_src=defaultdict(list)
    w2type_tgt=defaultdict(list)
    w2embeds_src=defaultdict(list)
    w2embeds_tgt=defaultdict(list)
    data_embed_src=np.load(data_embed_src,allow_pickle=True)[()]
    data_embed_tgt=np.load(data_embed_tgt,allow_pickle=True)[()]
    for line in open(datatext).readlines():
        line_key=produce_key(line)
        if line_key in data_embed_src and line_key in data_embed_tgt:
            assert len(data_embed_src[line_key])==len(line.split())==len(data_embed_tgt[line_key])
            for i,w in enumerate(line_key.split('\t')):
                if w in w2type_src:
                    continue
                if len(w2embeds_src[w])<max_example:
                    w2embeds_src[w].append(data_embed_src[line_key][i])
                    w2embeds_tgt[w].append(data_embed_tgt[line_key][i])
                else:
                    w2type_src[w]=np.mean(w2embeds_src[w],0)
                    w2type_tgt[w]=np.mean(w2embeds_tgt[w],0)
                    del w2embeds_src[w]
                    del w2embeds_tgt[w]
    for w in w2embeds_src:
        w2type_src[w]=np.mean(w2embeds_src[w],0)
        w2type_tgt[w]=np.mean(w2embeds_tgt[w],0)

    for w in w2type_src:
        outputdata_src.append(w2type_src[w])
        outputdata_tgt.append(w2type_tgt[w])
    print ('processed {0} embeddings',len(outputdata_src))
    return torch.from_numpy(np.vstack(outputdata_src)),torch.from_numpy(np.vstack(outputdata_tgt))
    
if __name__=='__main__':
    import argparse
    import os
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import defaultdict
    from trainer import Trainer
    from evaluation_script import evaluation
    from helpers import produce_key,normalize_embeddings
    import torch
    import numpy as np
    
    
    args = argparse.ArgumentParser('static transformation for cwe')
    args.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    args.add_argument('--data',type=str,help='training data file in txt')
    args.add_argument('--tgt_data', type=str, help='tgt data npy. tgt data is the space we will be test on.')
    args.add_argument('--src_data', type=str,help='src data in npy')
    args.add_argument('--test_data_dir', default='./eval_data',type=str, help='test data dir')
    args.add_argument('--norm', default='', type=str, help='normalize mode: normalize|center|normalize,center')
    args.add_argument('--save', action='store_true', default=False, help='save flag for model path: True or False')
    args.add_argument('--type',action="store_true",help='whether to use type level representations')
    args.add_argument('--eval',action='store_true',default=False,help='whether to perform evaluation')
    args.add_argument('--tgt_data_mean',default=None,help='in case of mean centering, the target data mean')
    args=args.parse_args()

    device = torch.device("cuda:{0}".format(args.gpu) if torch.cuda.is_available() and args.gpu>-1 else "cpu")
    print(device)

    # 1. data preprocessing
    if args.type:
        src_data,tgt_data=preprocess_data_type(args.data,args.src_data,args.tgt_data)
    else:
        src_data,tgt_data=preprocess_data(args.data,args.src_data,args.tgt_data)

    args.tgt_data_mean=normalize_embeddings(tgt_data,args.norm)
    args.src_data_mean=normalize_embeddings(src_data,args.norm)

    
    args.base_embed=os.path.basename(args.tgt_data).split('__')[1] # tgt embedding name
    args.src_embed=os.path.basename(args.src_data).split('__')[1] # src embedding name

    # 2. train
    trainer_ortho,trainer_mim=train(src_data,tgt_data,device)

    if args.save:
        torch.save(trainer_mim.model_tgt.state_dict(), args.base_embed+'..'+args.src_embed+'.pt')
        if args.tgt_data_mean is not None:        
            np.save(args.base_embed+'_mean_tgt.npy',np.array(args.tgt_data_mean.cpu()))

    # 3. evaluation
    if args.eval:
        evaluation(trainer_mim,args)
