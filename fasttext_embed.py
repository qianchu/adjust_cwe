
import numpy as np

def pad_vector(vector,dim):
    rep=int(dim/len(vector))
    vector=np.concatenate([vector for _ in range(rep)])
    if len(vector)<dim:
        pad_len=int(dim-len(vector))
        pad=np.zeros(pad_len)
        vector=np.concatenate([vector,pad])
 
    return vector

def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

def compile_model(model):
        model = open(model, errors='surrogateescape')
        words, x = read(model)
        word2ind = {word.lower(): i for i, word in enumerate(words)}
        model={w:x[i] for i,w in enumerate(words)}
        return model

def get_text2embed(text,model):
    text2embed={}
    with open (text,encoding='utf-8') as f:
        for line in f:
            line=line.split('\t')[0]
            key=produce_key(line.strip())
            line_vectors=[]
            line_lst=line.strip().split()
            found_unk=False
            for w in line_lst:
                w=w.lower()
                if w in model:
                    vector=model[w]
                else:
                    print ('unk found',w)
                    found_unk=True
                    if 'UNK' in model:
                        vector=model['UNK']
                    else:
                        vector=np.random.rand(args.dim)
                    # unks[w]=True
                    break
                if len(vector)<args.dim:
                    vector=pad_vector(vector,args.dim)

                line_vectors.append(vector)
            print (found_unk)
            if not found_unk:
                text2embed[key]=np.array(line_vectors,dtype='float32')
    print (len(text2embed))
    return text2embed


if __name__=='__main__':
    import argparse
    import numpy as np
    import os
    import random
    from sklearn.metrics.pairwise import cosine_similarity
    from helpers import *
    args = argparse.ArgumentParser('fasttext to vec in hdf5')
    args.add_argument('--model', type=str, help='fasttext model file')
    # args.add_argument('--model_type',default='fasttext',type=str,help='fasttext or w2v or glove')
    args.add_argument('--text', type=str, help='text file')
    args.add_argument('--dim',type=int, help='dimension size for ft embedding')
    args=args.parse_args()
    

    model=compile_model(args.model)
    if not args.dim:
        args.dim=len(model['the'])
    text2embed=get_text2embed(args.text,model)
    

    output_prefix=args.text+'__'+os.path.basename(args.model)+'.'+str(args.dim)
    
    np.save(output_prefix+'.ly_0__.hdf5.npy',text2embed)
            


