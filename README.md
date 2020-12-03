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
       --norm normalize,center 
```

The alignment matrix will be output as ./en_roberta-large~fasttext_wiki_en_1024_bin_200000_type0__mim.pt

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
    
