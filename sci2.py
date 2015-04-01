import os
import sys
import numpy
import pandas
import csv
import feature_reduction

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
from sklearn.pipeline import Pipeline
from sklearn import metrics

class StemTokenizer(object):
	def __init__(self):
		self.stemmer = EnglishStemmer()
	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in word_tokenize(doc)]

trainingData = pandas.DataFrame({'text': [], 'class': []})
with open('training.csv') as csvfile:
	reader = csv.DictReader(csvfile, delimiter=';')
	for row in reader:
		if row['polarity'] in ('0', '4'):
			trainingData = trainingData.append(pandas.DataFrame({'text': [feature_reduction.reduce(row['tweet'])], 'class': [row['polarity']]}, index=[row['id']]))

#stanford test
testData = pandas.DataFrame({'text': [], 'class': []})
with open('testing.csv') as csvfile:
	reader = csv.DictReader(csvfile, fieldnames=('polarity', 'id', 'date', 'query', 'user', 'tweet'))
	for row in reader:
		if row['polarity'] in ('0', '4'):
			testData = testData.append(pandas.DataFrame({'text': [feature_reduction.reduce(row['tweet'])], 'class': [row['polarity']]}, index=[row['id']]))

stop_words = stopwords.words('english')
stop_words.extend([feature_reduction.user_token.lower(), feature_reduction.url_token.lower()]);

pipe = Pipeline([
					('counter', CountVectorizer(ngram_range=(1, 1), stop_words=stop_words, tokenizer=StemTokenizer())), 
					#('normalizer', TfidfTransformer(smooth_idf=True, sublinear_tf=False, use_idf=True)), #should we want to use a TfidfTransformer
					#('classifier', MultinomialNB(fit_prior=False, alpha=1.0))
					#('classifier', SVC()) #default kernel is 'rbf', 'rbf' and 'poly' results in an error... why?
					#('classifier', SVC(kernel='poly')) #error...
					('classifier', SVC(kernel='linear')) #this one and the one below yield different results, why?
					#('classifier', LinearSVC())
					
				])

#applies fit and transform to first parameter at all stages but the last, where only fit is applied
pipe.fit(numpy.asarray(trainingData['text']), numpy.asarray(trainingData['class']))

#applies transform at all stages but the last one where predict is applied
predictions = pipe.predict(numpy.asarray(testData['text']))

#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html for details of categories
#total recall is the same value as the correctly predicted sentiments / all documents yields
print(metrics.classification_report(numpy.asarray(testData['class']), predictions, target_names= [ 'Positive', 'Negative'], digits = 5 ))