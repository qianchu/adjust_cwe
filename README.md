# adjust_cwe
This is the repository for the code in "Towards Better Context-aware Lexical Semantics:Adjusting Contextualized Representations through Static Anchors"

## Training Data
We trained the alignment from Wikipedia sentences. In the paper, we use all the contexts for the top 200k words. In practice, there is not much difference if you just use a smaller subset of Wikipedia. Here we release randomly selected 200k Wikipedia sentences as training data to construct the static anchors. We also precomputed the token-level representations of the wikipedia from contextualized models in npy files. 

You can download example data for aligning Roberta-large towards Fasttext from the links below:

[wikipedia data]: https://www.dropbox.com/s/tzun5ft47qx01g3/en_200k_shuffled.witespace.out_for_wa.en2en?dl=0

[wikipedia data word alignment file]: https://www.dropbox.com/s/s55sr7e8g2tkagj/en_200k_shuffled.witespace.out_for_wa.align.en2en?dl=0

[Roberta-large precomputed embeddings]: https://www.dropbox.com/s/1i5kkrwy9q6ilv4/wiki_roberta.zip?dl=0

[Fasttext precomputed embeddings]: https://www.dropbox.com/s/eyfpstqg6l59v05/wiki_fasttext.zip?dl=0

wikipedia 200k data: [wikipedia data]

wikipedia data word alignment file (this is automatically extracted as 0-0 1-1...): [wikipedia data word alignment file]

Roberta-large precomputed embeddings in npy file: [Roberta-large precomputed embeddings]

Roberta-large precomputed embeddings in npy file: [Fasttext precomputed embeddings]

## Evaluation data

[evaluation data]: https://www.dropbox.com/s/6ch7qykv71w3530/eval_data.zip?dl=0

We have processed evaluation data (CoSimlex, WiC, usim, and scws) ready to be downloaded from [evaluation data]

## To compute static anchors and to train the alignment, you can run the following:

python main.py  --align [wikipedia data word alignment file] --para [wikipedia data] --tgt_data [Roberta-large precomputed embeddings] --src_data [Fasttext precomputed embeddings] --cluster type --src_lg en --tgt_lg en --tgt --test_data_dir [evaluation data] --norm normalize,center 

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
    
