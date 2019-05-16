# Text Classification Example with Selected Newsgroups from Twenty Newsgroups

# Author: Thomas W. Miller (2019-03-08)
# Modified by: John Kiley (2019-05-04)

# Compares text classification performance under random forests
# Six vectorization methods compared:
#     TfidfVectorizer from Scikit Learn
#     CountVectorizer from Scikit Learn
#     HashingVectorizer from Scikit Learn
#     Doc2Vec from gensim (dimension 50)
#     Doc2Vec from gensim (dimension 100)
#     Doc2Vec from gensim (dimension 200)

# See example data and code from 
# https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

# The 20 newsgroups dataset comprises around 18000 newsgroups 
# posts on 20 topics split in two subsets: one for training (or development) 
# and the other one for testing (or for performance evaluation). 
# The split between the train and test set is based upon messages 
# posted before and after a specific date.

# =============================================================================
# Establish working environment
# =============================================================================

import multiprocessing
import re,string
import os
from pprint import pprint
import json

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer,\
    CountVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_decomposition import CCA  # canonical correlation
from sklearn.model_selection import train_test_split

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import nltk
from nltk.stem import PorterStemmer

# =============================================================================
# Set global variables
# =============================================================================

stoplist = nltk.corpus.stopwords.words('english')
DROP_STOPWORDS = False
SET_RANDOM = 9999
STEMMING = False  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 2  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH_LIST = [8, 16, 32, 64, 128, 256, 512]  # set vector length for TF-IDF and Doc2Vec
WRITE_VECTORS_TO_FILE = False

# JSON lines file for storing canonical correlatin results across many runs
cancor_results_file = open('cancor-results-file.jl', 'a+') # open new file or append to existing

#%%
# =============================================================================
# Utility Functions 
# =============================================================================

# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']    

# text parsing function for entire document string
def parse_doc(text):
    text = text.lower()
    text = re.sub(r'&(.)+', "", text)  # no & references  
    text = re.sub(r'pct', 'percent', text)  # replace pct abreviation  
    text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote 
    text = re.sub(r'[^\x00-\x7f]',r'', text)  # no non-ASCII strings    
    if text.isdigit(): text = ""  # omit words that are all digits    
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes  
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)        
    return text

# text parsing for words within entire document string
# splits the document string into words/tokens
# parses the words and then recreates a document string
# returns list of parsed words/tokens and parsed document string
def parse_words(text): 
    # split document into individual words
    tokens=text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out tokens that are one or two characters long
    tokens = [word for word in tokens if len(word) > 2]
    # filter out tokens that are more than twenty characters long
    tokens = [word for word in tokens if len(word) < 21]
    # filter out stop words if requested
    if DROP_STOPWORDS:
        tokens = [w for w in tokens if not w in stoplist]         
    # perform word stemming if requested
    if STEMMING:
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text 

#%%     
# =============================================================================
# Import data from JSON lines file
# =============================================================================

# identify directory JSON lines files 
docdir = r'C:\Users\johnk\Desktop\Grad School\6. Spring 2019\1. MSDS_453_NLP\6. Homework\week5\economics\econ\files'

print('\nList of file names in the data directory:\n')
print(os.listdir(docdir))

all_data = []

for file in os.listdir(docdir): 
    if file.endswith('.jl'):
        file_name = file.split('.')[0]  # keep name without extension
        with open(os.path.join(docdir,file), 'rb') as f:
            for line in f:
                all_data.append(json.loads(line))

#%%
# =============================================================================
# Unpack the list of dictionaries to create data frame
# =============================================================================

url = []
title = []
tags = []
text = []
labels = []
final_processed_tokens = []  # list of token lists for Doc2Vec
final_processed_text = [] # list of document strings for TF-IDF
labels = []  # use filenames as labels
for doc in all_data:
    url.append(doc['url'])
    title.append(doc['title'])
    tags.append(doc['tags'])
    labels.append(doc['labels'])
    text_string = doc['text']
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    text.append(text_string)
    final_processed_tokens.append(tokens)
    final_processed_text.append(text_string)

df = pd.DataFrame({"url": url,
                   "title": title,
                   "tags": tags,
                   "text": text,
                   "labels": labels},)

#the following is an example of what the processed text looks like.  
print('\nBeginning and end of the data frame:\n')
print(df.head(2))
print(df.tail(2))

#%%
# =============================================================================
# Split the corpus into training & testing sets
# =============================================================================

train_data, test_data = train_test_split(all_data, random_state=1)

#%%
# =============================================================================
# Preprocess the training set; set asside labels
# =============================================================================
train_titles = []
train_tokens = []  # list of token lists for gensim Doc2Vec
train_text = [] # list of document strings for sklearn TF-IDF
train_target = []  # use filenames as labels
for doc in train_data:
    train_titles.append(doc['title'])
    text_string = doc['text']
    train_target.append(doc['labels'])
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    train_tokens.append(tokens)
    train_text.append(text_string)
    
    
print('\nNumber of training documents:',
	len(train_text))	
#print('\nFirst item after text preprocessing, train_text[0]\n', 
#	train_text[0])
print('\nNumber of training token lists:',
	len(train_tokens))	
#print('\nFirst list of tokens after text preprocessing, train_tokens[0]\n', 
#	train_tokens[0])
#%%
# =============================================================================
# Spot check; confirm labels & titles match up
# =============================================================================

pprint(train_titles[:10])
pprint(train_target[:10])

#%%
# =============================================================================
# Preprocess the testing set; set asside labels
# =============================================================================
test_tokens = []  # list of token lists for gensim Doc2Vec
test_text = [] # list of document strings for sklearn TF-IDF
test_target= []  # use filenames as labels
test_titles = []

for doc in test_data:
    test_titles.append(doc['title'])
    text_string = doc['text']
    test_target.append(doc['labels'])
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    test_tokens.append(tokens)
    test_text.append(text_string)


print('\nNumber of testing documents:',
	len(test_text))	
#print('\nFirst item after text preprocessing, test_text[0]\n', 
#	test_text[0])
print('\nNumber of testing token lists:',
	len(test_tokens))	
#print('\nFirst list of tokens after text preprocessing, test_tokens[0]\n', 
#	test_tokens[0])
#%%
# =============================================================================
# Spot check; confirm labels & titles match up
# =============================================================================

pprint(test_titles[:10])
pprint(test_target[:10])

#%%

# =============================================================================
# Perform TFIDF & Word2Vec canonical correlation analysis
# =============================================================================
 
# create list for saving canonical correlation results
cancor_results = [] 

for VECTOR_LENGTH in VECTOR_LENGTH_LIST: 
    print('\n---------- VECTOR LENGTH ', str(VECTOR_LENGTH), ' ----------\n')
    # =============================================================================
    # TF-IDF
    # =============================================================================
    # note the ngram_range will allow you to include multiple-word tokens 
    # within the TFIDF matrix
    # Call Tfidf Vectorizer
    print('\nWorking on TF-IDF vectorization')
    Tfidf = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    	max_features = VECTOR_LENGTH)

    # fit the vectorizer using final processed documents.  
    TFIDF_matrix = Tfidf.fit_transform(final_processed_text)  

    tfidf_solution = pd.DataFrame(TFIDF_matrix.toarray())  # for modeling work  

    #creating datafram from TFIDF Matrix
    matrix = pd.DataFrame(TFIDF_matrix.toarray(), 
    	columns = Tfidf.get_feature_names(), 
    	index = labels)

    if WRITE_VECTORS_TO_FILE:
        tfidf_file_name = 'tfidf-matrix-'+ str(VECTOR_LENGTH) + '.csv'
        matrix.to_csv(tfidf_file_name)
        print('\nTF-IDF vectorization complete, matrix saved to ', tfidf_file_name, '\n')

    # =============================================================================
    # gensim Doc2Vec
    # =============================================================================
        
    print("\nWorking on Doc2Vec vectorization")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_processed_tokens)]
    model = Doc2Vec(documents, vector_size = VECTOR_LENGTH, window = 2, 
    	min_count = 1, workers = 4)

    doc2vec_df = pd.DataFrame()
    for i in range(0,len(final_processed_tokens)):
        vector = pd.DataFrame(model.infer_vector(final_processed_tokens[i])).transpose()
        doc2vec_df = pd.concat([doc2vec_df,vector], axis=0)

    doc2vec_solution = doc2vec_df  # for modeling work

    doc2vec_df = doc2vec_df.reset_index()

    doc_titles = {'title': labels}
    t = pd.DataFrame(doc_titles)

    doc2vec_df = pd.concat([doc2vec_df,t], axis=1)

    doc2vec_df = doc2vec_df.drop('index', axis=1)
    doc2vec_df = doc2vec_df.set_index('title')

    if WRITE_VECTORS_TO_FILE:
        doc2vec_file_name = 'doc2vec-matrix-'+ str(VECTOR_LENGTH) + '.csv'
        doc2vec_df.to_csv(doc2vec_file_name)
        print('\nDoc2Vec vectorization complete, matrix saved to ', doc2vec_file_name, '\n')

    # =============================================================================
    # Canonical Correlation... show relationship between TF-IDF and Doc2Vec
    # =============================================================================

    n_components = 3
    cca = CCA(n_components)
    cca.fit(X = tfidf_solution, Y = doc2vec_solution)

    U, V = cca.transform(X = tfidf_solution, Y = doc2vec_solution)

    for i in range(n_components):
        corr = np.corrcoef(U[:,i], V[:,i])[0,1]

    print('\nCanonical correlation betwen TF-IDF and Doc2Vec for vectors of length ', 
        str(VECTOR_LENGTH), ':', np.round(corr, 3), '\n')

    cancor_results.append(np.round(corr, 3))

    data = json.dumps({"STEMMING":STEMMING,
        "MAX_NGRAM_LENGTH":MAX_NGRAM_LENGTH,
        "VECTOR_LENGTH":VECTOR_LENGTH,
        "CANCOR":np.round(corr, 3)}) 
    cancor_results_file.write(data)
    cancor_results_file.write('\n')

print('\nSummary of Canonoical Correlation between TF-IDF and Doc2Vec Vectorizations\n')
print('\nVector Length Correlation')
print('\n-------------------------')
for item in range(len(VECTOR_LENGTH_LIST)):
    print('     ', VECTOR_LENGTH_LIST[item], '      ', cancor_results[item])

cancor_results_file.close()

#%%
# =============================================================================
# TF-IDF Vectorization
# =============================================================================

tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    max_features = VECTOR_LENGTH)
tfidf_vectors = tfidf_vectorizer.fit_transform(train_text)
print('\nTFIDF vectorization. . .')
print('\nTraining tfidf_vectors_training.shape:', tfidf_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use tfidf_vectorizer.transform, NOT tfidf_vectorizer.fit_transform
tfidf_vectors_test = tfidf_vectorizer.transform(test_text)
print('\nTest tfidf_vectors_test.shape:', tfidf_vectors_test.shape)
tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
tfidf_clf.fit(tfidf_vectors, train_target)
tfidf_pred = tfidf_clf.predict(tfidf_vectors_test)  # evaluate on test set
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))

#%%
# =============================================================================
# Count Vectorization
# =============================================================================

count_vectorizer = CountVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    max_features = VECTOR_LENGTH)
count_vectors = count_vectorizer.fit_transform(train_text)
print('\ncount vectorization. . .')
print('\nTraining count_vectors_training.shape:', count_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use count_vectorizer.transform, NOT count_vectorizer.fit_transform
count_vectors_test = count_vectorizer.transform(test_text)
print('\nTest count_vectors_test.shape:', count_vectors_test.shape)
count_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
count_clf.fit(count_vectors, train_target)
count_pred = count_clf.predict(count_vectors_test)  # evaluate on test set
print('\nCount/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, count_pred, average='macro'), 3))

#%%
# =============================================================================
# Hashing Vectorization
# =============================================================================

hashing_vectorizer = HashingVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    n_features = VECTOR_LENGTH)
hashing_vectors = hashing_vectorizer.fit_transform(train_text)
print('\ncount vectorization. . .')
print('\nTraining hashing_vectors_training.shape:', hashing_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use hashing_vectorizer.transform, NOT hashing_vectorizer.fit_transform
hashing_vectors_test = hashing_vectorizer.transform(test_text)
print('\nTest hashing_vectors_test.shape:', hashing_vectors_test.shape)
hashing_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
hashing_clf.fit(hashing_vectors, train_target)
hashing_pred = hashing_clf.predict(hashing_vectors_test)  # evaluate on test set
print('\nHashing/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, hashing_pred, average='macro'), 3))

#%%
# =============================================================================
# Doc2Vec
# =============================================================================

# =============================================================================
# Doc2Vec Vectorization (50 dimensions)
# =============================================================================

# doc2vec paper:  https://cs.stanford.edu/~quocle/paragraph_vector.pdf
#     has a neural net with 1 hidden layer and 50 units/nodes
# documentation at https://radimrehurek.com/gensim/models/doc2vec.html
# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
# tutorial on GitHub: 
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

print('\nBegin Doc2Vec Work')
cores = multiprocessing.cpu_count()
print("\nNumber of processor cores:", cores)

train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_tokens)]
# print('train_corpus[:2]:', train_corpus[:2])

# Instantiate a Doc2Vec model with a vector size with 50 words 
# and iterating over the training corpus 40 times. 
# Set the minimum word count to 2 in order to discard words 
# with very few occurrences. 
# window (int, optional) – The maximum distance between the 
# current and predicted word within a sentence.
print("\nWorking on Doc2Vec vectorization, dimension 50")
model_50 = Doc2Vec(train_corpus, vector_size = 50, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

model_50.train(train_corpus, total_examples = model_50.corpus_count, 
	epochs = model_50.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_50_vectors = np.zeros((len(train_tokens), 50)) # initialize numpy array
for i in range(0, len(train_tokens)):
    doc2vec_50_vectors[i,] = model_50.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_50_vectors.shape:', doc2vec_50_vectors.shape)
# print('doc2vec_50_vectors[:2]:', doc2vec_50_vectors[:2])

# vectorization for the test set
doc2vec_50_vectors_test = np.zeros((len(test_tokens), 50)) # initialize numpy array
for i in range(0, len(test_tokens)):
    doc2vec_50_vectors_test[i,] = model_50.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_50_vectors_test.shape:', doc2vec_50_vectors_test.shape)

doc2vec_50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
doc2vec_50_clf.fit(doc2vec_50_vectors, train_target) # fit model on training set
doc2vec_50_pred = doc2vec_50_clf.predict(doc2vec_50_vectors_test)  # evaluate on test set
print('\nDoc2Vec_50/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_50_pred, average='macro'), 3)) 

#%%

# =============================================================================
# Doc2Vec Vectorization (100 dimensions)
# =============================================================================
print("\nWorking on Doc2Vec vectorization, dimension 100")
model_100 = Doc2Vec(train_corpus, vector_size = 100, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

model_100.train(train_corpus, total_examples = model_100.corpus_count, 
	epochs = model_100.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_100_vectors = np.zeros((len(train_tokens), 100)) # initialize numpy array
for i in range(0, len(train_tokens)):
    doc2vec_100_vectors[i,] = model_100.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_100_vectors.shape:', doc2vec_100_vectors.shape)
# print('doc2vec_100_vectors[:2]:', doc2vec_100_vectors[:2])

# vectorization for the test set
doc2vec_100_vectors_test = np.zeros((len(test_tokens), 100)) # initialize numpy array
for i in range(0, len(test_tokens)):
    doc2vec_100_vectors_test[i,] = model_100.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_100_vectors_test.shape:', doc2vec_100_vectors_test.shape)

doc2vec_100_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
doc2vec_100_clf.fit(doc2vec_100_vectors, train_target) # fit model on training set
doc2vec_100_pred = doc2vec_100_clf.predict(doc2vec_100_vectors_test)  # evaluate on test set
print('\nDoc2Vec_100/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_100_pred, average='macro'), 3)) 

#%%
# =============================================================================
# Doc2Vec Vectorization (200 dimensions)
# =============================================================================
print("\nWorking on Doc2Vec vectorization, dimension 200")
model_200 = Doc2Vec(train_corpus, vector_size = 200, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

model_200.train(train_corpus, total_examples = model_200.corpus_count, 
	epochs = model_200.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_200_vectors = np.zeros((len(train_tokens), 200)) # initialize numpy array
for i in range(0, len(train_tokens)):
    doc2vec_200_vectors[i,] = model_200.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_200_vectors.shape:', doc2vec_200_vectors.shape)
# print('doc2vec_200_vectors[:2]:', doc2vec_200_vectors[:2])

# vectorization for the test set
doc2vec_200_vectors_test = np.zeros((len(test_tokens), 200)) # initialize numpy array
for i in range(0, len(test_tokens)):
    doc2vec_200_vectors_test[i,] = model_200.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_200_vectors_test.shape:', doc2vec_200_vectors_test.shape)

doc2vec_200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
doc2vec_200_clf.fit(doc2vec_200_vectors, train_target) # fit model on training set
doc2vec_200_pred = doc2vec_200_clf.predict(doc2vec_200_vectors_test)  # evaluate on test set
print('\nDoc2Vec_200/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_200_pred, average='macro'), 3)) 

#%%
# =============================================================================
# Print Results
# =============================================================================

print('\n\n------------------------------------------------------------------------')
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))
print('\nCount/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, count_pred, average='macro'), 3))
print('\nHashing/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, hashing_pred, average='macro'), 3))
print('\nDoc2Vec_50/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_50_pred, average='macro'), 3)) 
print('\nDoc2Vec_100/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_100_pred, average='macro'), 3))   
print('\nDoc2Vec_200/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_200_pred, average='macro'), 3)) 
print('\n------------------------------------------------------------------------')