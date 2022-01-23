# adjust_cwe
This is the repository for the code in "Towards Better Context-aware Lexical Semantics:Adjusting Contextualized Representations through Static Anchors"

## Encode train and evaluation data
 ```
 bash encode.sh [cuda] [model] [layer] [static model] [dimension]
 ```
 [static model] is the text file of a static model's embeddings. For example, you can download the fasttext model [here](https://www.dropbox.com/s/8uru7gp9ipo24p6/wiki.en.1024.vec?dl=0). 
 [layer] is the range of layers from the contextual representations to take an average
 
 ```
 bash encode.sh 0 bert-large-cased 12-25 fasttext-wiki-en-1024-bin-vec 768 &> encode.log &
 ```
## Train the mapping

```
bash train.sh [train data] [target data] [source data]
```
[train data] is the text file of the train data. [target data] and [source data] are the npy embeddings precomputed for [train data]. 

EG.
```
bash train.sh 1 ./train_data/en_200k_shuffled.whitespace.txt ./train_data/en_200k_shuffled.whitespace.txt__bert-large-cased.ly_12-25__.hdf5.npy ./train_data/en_200k_shuffled.whitespace.txt__wiki.en.768.bin.vec.1024.ly_0__.hdf5.npy
```


The alignment matrix will be output as 'bert-large-cased.ly_12-25..wiki.en.1024.vec.1024.ly_0.pt'

If you chose centering in the preprocessing step, the mean numpy array for the target embeddings (which is BERT in this example) will be stored in 'bert-large-cased.ly_12-25_mean_tgt.npy'. The mean numpy array for the source embeddings (which is Fasttext in this example) will be stored in 'en_roberta\-large\~fasttext_wiki_en_1024_bin_200000_type0__mean_src.npy'.


## About the test results
usim and scws results are printed when you run main.py

Test predictions for CoSimlex can be found in:  

    before alignment: eval_data/bert-large-cased.ly_12-25

    after alignment: eval_data/bert-large-cased.ly_12-25.adjusted

    You can then submit the test predictions to https://competitions.codalab.org/competitions/20905

Test predictions for WiC can be found in:

    before alignment: eval_data/WiC_dataset/test/bert-large-cased.ly_12-25

    after alignment: eval_data/WiC_dataset/test/bert-large-cased.ly_12-25.adjusted

    You can then submit the test predictions to https://competitions.codalab.org/competitions/20010
    
## Apply the learned mapping on your own representations

```
import numpy as np
from trainer import Trainer
from helpers import normalize_embeddings
from extract_features import examples2embeds
import torch
from transformers import AutoModel, AutoTokenizer


class LinearProjection(torch.nn.Module):
    def __init__(self, D_in,  D_out):
        """
        In the constructor we instantiate a nn.Linear modules and assign them as
        member variables.
        """
        super(LinearProjection, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out,bias=False)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        project=self.linear(x)
        return project

###1. initialization the mapping
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
mean='bert-base-uncased.ly_9-13_mean_tgt.npy'
mapping_path='bert-large-cased.ly_12-25..wiki.en.1024.vec.1024.ly_0.pt'
dim=1024
mapping = LinearProjection(dim, dim).to(device)
mapping.load_state_dict(torch.load(mapping_path))

###2.  get torch embeddings from CWE eg. from huggingface. Below is a random initialization
tokenizer=AutoTokenizer.from_pretrained('bert-large-cased')
model=AutoModel.from_pretrained('bert-large-cased',output_hidden_states=True,output_attentions=True)
model.to(device)
layers='12-25'
max_seq_length=128
examples=['this is the first example .','this is the second example .']
#examples= [example.split() for example in examples]
embs=examples2embeds(examples,tokenizer,model,device,max_seq_length,layers,lg=None)
print (embs)
#embs now is a list of arrays each of which consists of word-level embeddings in an example. Say that we want to apply the mapping on the first word's embedding in the second example, we can extract it as:
emb=torch.tensor(embs[1][0:1])

###3. apply preprocessing
## normalize and center    
#normalize_embeddings(emb,'normalize,center',mean=mean)
# only normalize
normalize_embeddings(emb,'normalize')
## or no preprocessing at all

###4. apply mapping
mapped=mapping(emb.to(device))


```

