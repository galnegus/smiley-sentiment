import os
import sys
import numpy
#from pandas import DataFrame #allows us to address DataFrame by DataFrame only...
#the 2 following forces us to address DataFrame by pandas.DataFrame
#import pandas.DataFrame
import pandas
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

trainingData = pandas.DataFrame({'text': [], 'class': []})
with open('training.csv') as csvfile:
	reader = csv.DictReader(csvfile, delimiter=';')
	for row in reader:
		if row['polarity'] in ('0', '4'):
			trainingData = trainingData.append(pandas.DataFrame({'text': [row['tweet']], 'class': [row['polarity']]}, index=[row['id']]))

#stanford test
stanford_test = []
testData = pandas.DataFrame({'text': [], 'class': []})
with open('testing.csv') as csvfile:
	reader = csv.DictReader(csvfile, fieldnames=('polarity', 'id', 'date', 'query', 'user', 'tweet'))
	for row in reader:
		if row['polarity'] in ('0', '4'):
			#stanford_test.append((row['tweet'], row['polarity']))
			testData = testData.append(pandas.DataFrame({'text': [row['tweet']], 'class': [row['polarity']]}, index=[row['id']]))


#our 2 classifications
POSITIVE = "4"
NEGATIVE = "0"
	
trainingData = trainingData.reindex(numpy.random.permutation(trainingData.index)) #don't know why this reindexing is important...

count_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word') #how many times a certain word has to appear in the text to take it into account
counts = count_vectorizer.fit_transform(numpy.asarray(trainingData['text']))

classifier = MultinomialNB()
#print(counts)
targets = numpy.asarray(trainingData['class'])
classifier.fit(counts, targets)

#examples = ['linux viagra', "I'm going to attend the Linux users group tomorrow."]
#examples = ['#summer is coming! yay!', "I'm so goddamn angry right now, why is @apple introducing this new feature?"]
#examples = csv_test_feed(testing_file, 6, 5, ',')

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