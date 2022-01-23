# SemEval 2020 Task 3
## Predicting the (Graded) Effect of Context in Word Similarity

This package contains the test data for the SemEval 2020 task 3 for the English language. 

The task is divided in two different subtasks:

	1. Subtask1: Predicting the degree and direction of change in the human annotator's scores of similarity when presented with the same pair of words within two different contexts. This task directly addresses our main question. It evaluates how well systems are able to model the effect that context has in human perception of similarity.
	2. Subtask2: Predicting the human scores of similarity for a pair of words within the same two contexts. This is a more traditional task which evaluates systems' ability to model both similarity of words and the effect that context has on it.

## Evaluation Metrics

For Subtask 1, in which we ask participants to predict the change in similarity scores between the two contexts, the sign of the results is very important as it determines the direction of change predicted. In this case we use the uncentered variation of the Pearson correlation, calculated using the standard deviation from zero rather than from the mean value. 

For Subtask2, following the example of 'SemEval 2017 Task 2: Multilingual and Cross-lingual Semantic Word Similarity' (Camacho-Collados et al., 2017) we are scoring the results using the harmonic mean of the Pearson and Spearman correlations.

## Annotation and Test Data
	
You can see instructions that were provided to the annotators and an example of how the survey looked like in the included pdf: **instructions-survey_example.pdf**

The test data contains contains:

	- 340 English pairs: data_en.tsv
	- 24 Finnish pairs: data_fi.tsv
	- 112 Croatian pairs: data_hr.tsv
	- 111 Slovene pairs: data_sl.tsv

In addition to this README file the package contains five different folders:

- data:
	
	Tab separated files containing the datasets:

	**word1** \<tab> **word2** \<tab> **context1** \<tab> **context2** \<tab> **word1_context1** \<tab> **word2_context1** \<tab> **word1_context2** \<tab> **word2_context2**

	The additional fields contain the 'inflected' versions of the words as they appear in each of the contexts.
	For all languages the target words are additionally marked with a \<strong>\</strong>.

- gold:

	The gold standard human scores.
	These annotations won't be released until the end of the evaluation period.

	**sim_context1** \<tab> **sim_context2** \<tab> **change** = (sim_score_context2 - sim_score_context1)

- submission_examples:

	Zip files containing different submissions that you can upload to Codalab to test the process.

- res1:
	
	Example of system generated results (multilingual Bert) which serve as baselines.
	Please make sure you follow the exact same format, names for the files and the headers inside them.
	One tsv formated file per language, all compressed in one zip file. 
	You can submit just the languages in which you want to participate, however the submission of a result for the English language is mandatory.

	We will declare a winner per each of the languages, unfortunately we weren't able to create independent leader boards per language at Codalab, so for the purpose of the leader board there, we will order submissions based on the English results.

	Each file contains only one column, with the header 'change', containing the difference between the similarity scoring within each of the contexts:

	**change** = (sim_score_context2 - sim_score_context1)
	

- res2:
	
	Example of system generated results (multilingual Bert) which serve as baseline.
	Please make sure you follow the exact same format, names for the files and the headers inside them.
	Same conditions apply and same policy will rule the order of the Codalab leader board as in subtask1.

	Each file contains two columns with the similarity score per context: 

	**sim_context1** \<tab> **sim_context2**



Contact:

	- Carlos Santos Armendariz (c.santosarmendariz@qmul.ac.uk)
	- Matthew Purver (m.purver@qmul.ac.uk)