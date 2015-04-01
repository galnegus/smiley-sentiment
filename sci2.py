import os
import sys
import numpy
import pandas
import csv
import feature_reduction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import EnglishStemmer

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

class StemTokenizer(object):
	def __init__(self):
		self.stemmer = EnglishStemmer()
	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in word_tokenize(doc)]

stop_words = stopwords.words('english')
stop_words.extend([feature_reduction.user_token.lower(), feature_reduction.url_token.lower()]);
count_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words, tokenizer=StemTokenizer()) #how many times a certain word has to appear in the text to take it into account
counts = count_vectorizer.fit_transform(numpy.asarray(trainingData['text']))

classifier = MultinomialNB(fit_prior=False, alpha=1.0)

targets = numpy.asarray(trainingData['class'])
classifier.fit(counts, targets)

example_counts = count_vectorizer.transform(numpy.asarray(testData['text']))
predictions = classifier.predict(example_counts)
print("Processed predictions: " + str(len(predictions)))
count = 0
for (i, sentiment) in enumerate(numpy.asarray(testData['class'])):
	#print(str(predictions[i]) + ": " + sentiment)
	if predictions[i] == sentiment:
		count = count + 1

print("count: " + str(count))
print("predictions: " + str(len(predictions)))
print(count / float(len(predictions)))
#print(predictions)