# adjust_cwe
This is the repository for the code in "Towards Better Context-aware Lexical Semantics:Adjusting Contextualized Representations through Static Anchors"

## Training Data
In the paper, we train the static anchors from top most frequent 200k words using the Wikipedia dump. In practice, there is not much difference if you just use a smaller subset of Wikipedia. Here we release randomly selected 200k Wikipedia sentences as training data to construct the static anchors. 

You can download example data from the links below:

[wikipedia data]: https://www.dropbox.com/s/tzun5ft47qx01g3/en_200k_shuffled.witespace.out_for_wa.en2en?dl=0

[wikipedia data word alignment file]: https://www.dropbox.com/s/s55sr7e8g2tkagj/en_200k_shuffled.witespace.out_for_wa.align.en2en?dl=0

[Roberta-large]: https://www.dropbox.com/s/1i5kkrwy9q6ilv4/wiki_roberta.zip?dl=0

[Fasttext]: https://www.dropbox.com/s/eyfpstqg6l59v05/wiki_fasttext.zip?dl=0

[Bert-large-cased]: https://www.dropbox.com/s/mre7501y6fyx4vz/en_200k_shuffled.witespace.bert-large-cased.ly-12.hdf5.npy?dl=0

wikipedia 200k data: [wikipedia data]

wikipedia data word alignment file (this is automatically extracted as 0-0 1-1...): [wikipedia data word alignment file]

Roberta-large precomputed token-level embeddings in npy file: [Roberta-large]

Fasttext precomputed embeddings in npy file: [Fasttext]

Bert-large-cased precomputed embeddings in npy: [Bert-large-cased]

Alternatively, you can extract features using code from: https://github.com/qianchu/transformers/

## Evaluation data

[evaluation data]: https://www.dropbox.com/s/q6l6l0gjig1abqc/eval_data.zip?dl=0

We have processed evaluation data (CoSimlex, WiC, usim, and scws) ready to be downloaded from [evaluation data]

## To compute static anchors and to train the alignment
You can run the following to train the alignment from Roberta-large to Fasttext, and evaluate on the transformed Roberta-large space. 

```python
python main.py  \
    --align [wikipedia data word alignment file] \
       --para [wikipedia data] \
       --tgt_data [Roberta-large] \
       --src_data [Fasttext] \
       --cluster type \
       --src_lg en \
       --tgt_lg en \
       --tgt \
       --test_data_dir [evaluation data] \
       --norm normalize,center \
       --eval \
```

The alignment matrix will be output as './en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mim.pt'

If you chose centering in the preprocessing step, the mean numpy array for the target embeddings (which is Roberta in this example) will be stored in 'en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mean_tgt.npy'. The mean numpy array for the source embeddings (which is Fasttext in this example) will be stored in 'en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mean_src.npy'.


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

