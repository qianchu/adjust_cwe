# adjust_cwe
This is the repository for the code in "Towards Better Context-aware Lexical Semantics:Adjusting Contextualized Representations through Static Anchors"

## Encode train and evaluation data
 ```
 bash encode.sh [cuda] [model] [layer] [static model] [dimension]
 ```
 [static model] is the text file of a static model's embeddings. You can download the fasttext model that we used in our experiments [here]
 ```
 bash encode.sh 0 bert-large-cased 12-25 fasttext-wiki-en-1024-bin-vec 768 &> encode.log &
 ```
## Train the mapping

```
bash train.sh [train data] [target data] [source data]
```
[train data] is the text train data

EG.
```
bash train.sh 1 ./train_data/en_200k_shuffled.whitespace.txt ./train_data/en_200k_shuffled.whitespace.txt__bert-large-cased.ly_12-25__.hdf5.npy ./train_data/qianchu/adjust_cwe/en_200k_shuffled.witespace.txt__fasttext-wiki-en-1024-bin-vec.ly_0-0__.hdf5.npy
```


The alignment matrix will be output as './en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mim.pt'

If you chose centering in the preprocessing step, the mean numpy array for the target embeddings (which is Roberta in this example) will be stored in 'en_roberta\-large\~fasttext_wiki_en_1024_bin_200000_type0__mean_tgt.npy'. The mean numpy array for the source embeddings (which is Fasttext in this example) will be stored in 'en_roberta\-large\~fasttext_wiki_en_1024_bin_200000_type0__mean_src.npy'.


## About the test results
usim and scws results are printed when you run main.py

Test predictions for CoSimlex can be found in:  

    before alignment: [evaluation data]/context_simlex/en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0

    after alignment: [evaluation data]/context_simlex/en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mim

    You can then submit the test predictions to https://competitions.codalab.org/competitions/20905

Test predictions for WiC can be found in:

    before alignment: [evaluation data]/eval_data/WiC_dataset/test/en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0

    after alignment: [evaluation data]/eval_data/WiC_dataset/test/en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0_mim

    You can then submit the test predictions to https://competitions.codalab.org/competitions/20010
    
## Apply the learned mapping on your own representations

```
import numpy as np
from trainer import Trainer
from helpers import normalize_embeddings
import torch


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

###1. initialization
mean=torch.from_numpy(np.load('en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mean_tgt.npy'))
mapping_path='en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mim.pt'
dim=1024
mapping = LinearProjection(dim, dim)
mapping.load_state_dict(torch.load(mapping_path))

###2.  get torch embeddings from CWE eg. from huggingface. Below is a random initialization
emb=torch.rand(1,dim)

###3. apply preprocessing
# normalize and center    
normalize_embeddings(emb,'normalize,center',mean=mean)
## only normalize
# normalize_embeddings(emb,'normalize')
## or no preprocessing at all

###4. apply mapping
mapped=mapping(emb)
print (mapped)

```

