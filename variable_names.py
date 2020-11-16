UNIDIRECT_MODEL='unidirection'
BIDIRECT_MODEL='bidirection'
MUSE='muse'
EXACT='exact'
EXACT_ORTHO='exact_ortho'
NORMALIZE='normalize'
CENTER='center'
MIM='mim'

NP_BDI='nonparallel_bdi'
NP_BDI_SOURCE='nonparallel_bdi_source'
P_BDI='parallel_bdi'
NP_CBDI='nonparallel_cbdi'
P_CBDI='parallel_cbdi'
NP_CLS='nonparallel_crosslingual_lexical_substitution'
NP_CLS_type='nonparallel_crosslingual_lexical_substitution_type'
NP_CLS_WSD='nonparallel_crosslingual_lexical_substitution_wsd'
NP_CLS_WSD_type='nonparallel_crosslingual_lexical_substitution_wsd_type'
NP_CBDI_200K='nonparallel_cbdi.200k'
P_CBDI_200K='parallel_cbdi.200k'
NP_CBDI_TOK_CONTEXT_AVG=NP_CBDI+'__token_context_avg'
NP_CBDI_TYPE_CONTEXT_AVG=NP_CBDI+'__type_context_avg'
NP_CBDI_TYPE_CONTEXT_TGT_AVG=NP_CBDI+'__type_context_tgt_avg'
P_CBDI_TYPE_CONTEXT_AVG=P_CBDI+'__type_context_avg'
P_CBDI_TOK_CONTEXT_AVG=P_CBDI+'__token_context_avg'
P_CBDI_TYPE_CONTEXT_TGT_AVG=P_CBDI+'__type_context_tgt_avg'
LEXSUB_EN='lexsub_en'
LEXSUB_CL='lexsub_cl'
BDI_LONGTAIL_CRAWL='bdi_longtail_CRAWL'
BDI_LONGTAIL_GENERAL='bdi_longtail_GENERAL'
BDI_LONGTAIL_HIML='bdi_longtail_HIML'
BDI_LONGTAIL_NEWS='bdi_longtail_NEWS'
BDI_LONGTAIL_RAREHIML='bdi_longtail_RAREHIML'
BDI_MUSE_DE='bdi_muse_de'
BDI_MUSE='bdi_muse'
BDI_LONGTAIL_CRAWL_1='bdi_longtail_CRAWL_1'
BDI_LONGTAIL_GENERAL_1='bdi_longtail_GENERAL_1'
BDI_LONGTAIL_HIML_1='bdi_longtail_HIML_1'
BDI_LONGTAIL_NEWS_1='bdi_longtail_NEWS_1'
BDI_LONGTAIL_RAREHIML_1='bdi_longtail_RAREHIML_1'
BDI_LONGTAIL_CRAWL_5='bdi_longtail_CRAWL_5'
BDI_LONGTAIL_GENERAL_5='bdi_longtail_GENERAL_5'
BDI_LONGTAIL_HIML_5='bdi_longtail_HIML_5'
BDI_LONGTAIL_NEWS_5='bdi_longtail_NEWS_5'
BDI_LONGTAIL_RAREHIML_5='bdi_longtail_RAREHIML_5'
BDI_LONGTAIL_MT='bdi_longtail_mt'


BDI_LONGTAILS=[BDI_LONGTAIL_CRAWL,BDI_LONGTAIL_GENERAL,BDI_LONGTAIL_NEWS,BDI_LONGTAIL_RAREHIML,BDI_LONGTAIL_HIML]
BDI_LONGTAILS_1=[bdi+'|1' for bdi in BDI_LONGTAILS]
BDI_LONGTAILS_5=[bdi+'|5' for bdi in BDI_LONGTAILS]
BDI_LONGTAILS_all=[bdi+'|all' for bdi in BDI_LONGTAILS]













MODEL2FNAME={}
MODEL2FNAME['bert-base-cased']={'zh':'bert-base-chinese.ly-12','en':'bert-base-cased.ly-12'}
# MODEL2FNAME['bert_uncased']={'en':'bert-base-uncased.ly-12'}
MODEL2FNAME['bert-large-cased']={'en':'bert-large-cased.ly-12'}
MODEL2FNAME['xlm-roberta-large']={ lg:'xlm-roberta-large.ly-12' for lg in ['en','zh']}
MODEL2FNAME['ft']={'en':'fasttext_wiki_en_1024_bin.ly-0'}
MODEL2FNAME['glove']={'en':'glove_en_glove_300_txt.ly-0'}
MODEL2FNAME['w2v']={'en':'w2v_en_w2v_txt.ly-0'}

MODEL2FNAME['roberta-large']={'en':'roberta-large.ly-12'}
MODEL2FNAME['roberta-large-ly2']={'en':'roberta-large.ly-2'}

MODEL2FNAME['xlnet-large-cased']={'en':'xlnet-large-cased.ly-23'}
MODEL2FNAME['t5-large']={'en':'t5-large.ly-12'}
MODEL2FNAME['bert-large-cased-whole-word-masking']={'en':'bert-large-cased-whole-word-masking.ly-12'}


MODEL2FNAME['bert_multi']={lg:'bert-base-multilingual-cased.ly-12' for lg in ['en','zh','es','de']}
MODEL2FNAME['elmo']={'zh':'chinese_elmo.ly-1', 'en': 'allennlp_english_elmo.ly-1', 'es': 'spanish_elmo.ly-1'}
MODEL2FNAME['fasttext']={'zh':'wiki.zh.300.bin','en':'wiki.en.300.bin','es':'wiki.es.300.bin','de':'wiki.de.300.bin'}
MODEL2FNAME['fasttext_joint']={lg:'wiki.en-de.300.bin' for lg in ['en','de']}
MODEL2FNAME['glove_en_glove_300_txt']={'en':'glove_en_glove_300_txt.ly-0'}
MODEL2FNAME['w2v_en_w2v_txt']={'en':'w2v_en_w2v_txt.ly-0'}
MODEL2FNAME['cwetype_en_roberta_type_txt']={'en':'cwetype_en_roberta_type_txt.ly-0'}
MODEL2FNAME['cwetype_en_roberta_type_lemma_txt']={'en':'cwetype_en_roberta_type_lemma_txt.ly-0'}
MODEL2FNAME['bert_indie_multi']={'zh':'bert-base-multilingual-cased.ly-12','es':'bert-base-multilingual-cased.ly-12','de':'bert-base-multilingual-cased.ly-12','en':'bert-base-cased.ly-12'}
MODEL2FNAME['fasttext_wiki_en_1024_bin']={'en':'fasttext_wiki_en_1024_bin.ly-0'}
lg2wn={'en':'en','es':'spa','zh':'ctn'}
lg2corpus={'de':{'en':'europarl-v7.de-en.en.tc_only.tokenized.parallel','de':'europarl-v7.de-en.de.tc_only.tokenized.parallel'},'zh':{'en':'en.txt.parallel','zh':'ch_tra.txt.parallel'},'es':{lg:'wmt13_{0}_corpus.tokenized.parallel'.format(lg) for lg in ['en','es'] }}
lg2vocab={'zh':{'zh':'ch_vocab_20k', 'en': 'en_vocab_20k'}, 'es':{lg: "en_es.{0}.vocab_20k".format(lg) for lg in ['en','es']}}

EVAL_DATA={}