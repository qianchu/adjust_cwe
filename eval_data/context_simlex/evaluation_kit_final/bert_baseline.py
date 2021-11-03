from bert_embedding import BertEmbedding
from scipy import spatial
import csv
import sys
from unidecode import unidecode

## This script makes use of an open source project to generate token embeddings with multilingual Bert
## The project was created by Gary Lai (https://gary-lai.com/) and is available in github:
## https://github.com/imgarylai/bert-embedding

def unbolded(context):
#     context = str.replace(context, '-', '')
    context = str.replace(context, '<strong>', '')
    return str.replace(context, '</strong>', '')

## Load bert model
bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_uncased', max_seq_length=230)

languages = ('en', 'fi', 'hr', 'sl')

for lan in languages:

    rows = []
    similarities = []

    print(f'\nLANGUAGE: {lan.upper()}\n')

    with open(f'./data/data_{lan}.tsv', 'r') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for index, row in enumerate(csvreader):

            print(f"{index} {row['word1']}-{row['word2']}")

            ## Results contains several results for several sentences
            results = bert_embedding([unbolded(row['context1'])])
            ## Result here contains the results for first sentence
            result = results[0]
            ## result[0] is a list with all the tokens
            ## result[1] is a list with the embeddings per token
            tokens = result[0]
            embeddings = result[1]

            # print(tokens)
            
            i = 0
            for token, embedding in zip(tokens, embeddings):
                if unidecode(token) == unidecode(row['word1_context1'].lower()):
                    word1_context1 = embedding
                    i += 1
                    break

            for token, embedding in zip(tokens, embeddings):
                if unidecode(token) == unidecode(row['word2_context1'].lower()):
                    word2_context1 = embedding
                    i += 1
                    break
                    
            if i != 2:
                print(f'i = {i}')
                print(f'Word1_context1 = {unidecode(row["word1_context1"].lower())}')
                print(f'Word2_context1 = {unidecode(row["word2_context1"].lower())}')
                print(tokens)
                sys.exit(f'Error: Not able to find the words in context: {row["context1"]}')

            sim_context1 = 1 - spatial.distance.cosine(word1_context1, word2_context1)

            ## Results contains several results for several sentences
            results = bert_embedding([unbolded(row['context2'])])
            ## Result here contains the results for first sentence
            result = results[0]
            ## result[0] is a list with all the tokens
            ## result[1] is a list with the embeddings per token
            tokens = result[0]
            embeddings = result[1]

            # print(tokens)
            
            i = 0
            for token, embedding in zip(tokens, embeddings):
                if unidecode(token) == unidecode(row['word1_context2'].lower()):
                    word1_context2 = embedding
                    i += 1
                    break

            for token, embedding in zip(tokens, embeddings):
                if unidecode(token) == unidecode(row['word2_context2'].lower()):
                    word2_context2 = embedding
                    i += 1
                    break
            
            if i != 2:
                print(f'i = {i}')
                print(f'Word1_context2 = {unidecode(row["word1_context2"].lower())}')
                print(f'Word2_context2 = {unidecode(row["word2_context2"].lower())}')
                print(tokens)
                sys.exit(f'Error: Not able to find the words in context: {row["context2"]}')

            sim_context2 = 1 - spatial.distance.cosine(word1_context2, word2_context2)

            similarities.append([sim_context1, sim_context2])

    columns = ['change']
    with open(f'./res1/results_subtask1_{lan}.tsv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
        csvwriter.writerow(columns)
        for sim in similarities:
            change = sim[1] - sim[0]
            csvwriter.writerow([change])

    columns = ['sim_context1', 'sim_context2']
    with open(f'./res2/results_subtask2_{lan}.tsv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
        csvwriter.writerow(columns)
        for sim in similarities:
            csvwriter.writerow(sim)
