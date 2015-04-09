import os
import csv
import feature_reduction
import data_sets
import argparse

#preprocessing
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import EnglishStemmer

#classifiers and classifier tools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

class StemTokenizer(object):
	def __init__(self):
		self.stemmer = EnglishStemmer()
	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in word_tokenize(doc)]
		
#set up command line parser
parser = argparse.ArgumentParser(description='Run a classifier session with training, testing and output of result statistics.')
parser.add_argument('training_set', choices=['sts_2000', 'sf_2000', 'sf_10000', 'sf_100000', 'sf_1600000'], action='store', help='the training set to use for this run')
parser.add_argument('classifier', action='store', choices=['nb', 'svm_poly', 'svm_linear', 'me'], help='the classifier to use for this run')
parser.add_argument('-grams', action="store", default='uni', choices=['uni', 'bi', 'both'], help='whether we want to use unigrams, bigrams or both, stopwords may be problematic if used with other than unigrams')
parser.add_argument('--stopwords', action='store_true', default=False, help='if we want to use stopwords for this run or not')
parser.add_argument('--tfid', action='store_true', default=False, help='if we want data to be processed by a normalizer in this run')

#parse arguments
args = parser.parse_args()

#set up the run
trainingData = getattr(data_sets, args.training_set)()
testData = data_sets.testing()
stop_words = stopwords.words('english')
stop_words.extend([feature_reduction.user_token.lower(), feature_reduction.url_token.lower()]);

#setup parameters for our classifier pipeline
params = []
grams = (1,1)
if(args.grams == 'bi'): grams = (2,2)
elif(args.grams == 'both'): grams = (1,2)
if(args.stopwords == True): params.append(('counter', CountVectorizer(ngram_range=grams, stop_words=stop_words, tokenizer=StemTokenizer())))
else: params.append(('counter', CountVectorizer(ngram_range=grams, tokenizer=StemTokenizer())))
if(args.tfid == True): params.append(('normalizer', TfidfTransformer(smooth_idf=False, sublinear_tf=True, use_idf=True))) #should we want to use a TfidfTransformer
if(args.classifier == 'nb'): params.append(('classifier', MultinomialNB(fit_prior=False, alpha=1.0)))
elif(args.classifier == 'svm_poly'): params.append(('classifier', SVC(kernel='poly', gamma=1.0, coef0=21))) #error...
elif(args.classifier == 'svm_linear'): params.append(('classifier', LinearSVC(loss='hinge', C=8, fit_intercept=False))) #use wout params if not stopwords and tfid, similar to SVC(kernel='linear') but implemented differently and should be more accurate
elif(args.classifier == 'me'): params.append(('classifier', LogisticRegression(C=5.5))) #maximum entropy

pipe = Pipeline(params)

#applies fit and transform to first parameter at all stages but the last, where only fit is applied
pipe.fit(trainingData['text'], trainingData['class'])

#applies transform at all stages but the last one where predict is applied
predictions = pipe.predict(testData['text'])

#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html for details of categories
#total recall is the same value as the correctly predicted sentiments / all documents yields
print(metrics.classification_report(testData['class'], predictions, target_names= ['Positive', 'Negative'], digits = 5 ))