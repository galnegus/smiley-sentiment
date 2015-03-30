import os
import sys
import numpy
#from pandas import DataFrame #allows us to address DataFrame by DataFrame only...
#the 2 following forces us to address DataFrame by pandas.DataFrame
#import pandas.DataFrame
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def csv_training_feed(path, cols):
	i = 0
	if os.path.isfile(path):
		f = open(path)
		for line in f:
			row_tokens = line.split(",", cols - 1)
			if len(row_tokens) == cols:
				i += 1
				yield row_tokens #return one line from the input
		print("Processed training data: " + str(i))
		f.close()

def csv_test_feed(path, cols, col_to_get, delimiter):
	if os.path.isfile(path):
		f = open(path)
		lines = []
		for line in f:
			row_tokens = line.split(delimiter, cols - 1)
			if len(row_tokens) == cols:
				lines.append(row_tokens[col_to_get].rstrip(' "').lstrip(' "'))
		f.close()
		return lines
  
#check command line arguments...
if len(sys.argv) != 3:
	sys.exit("Please supply input and output file arguments")
	
#our 2 classifications
POSITIVE = 0
NEGATIVE = 1

data = pandas.DataFrame({'text': [], 'class': []})
training_file = sys.argv[1]
testing_file = sys.argv[2]
for arr in csv_training_feed(training_file, 5):
	classification = -1 #only care about those that has either a positive or negative classification, no irrelevant ones
	arr[1] = arr[1].lstrip(' "').rstrip(' "').lower()
	if arr[1] == "negative":
		classification = NEGATIVE
	elif arr[1] == "positive":
		classification = POSITIVE
	if classification != -1: #this is a message that we are interested in...
		data = data.append(pandas.DataFrame({'text': [arr[4].lstrip(' "').rstrip(' "\n')], 'class': [classification]}, index=[arr[2].lstrip(' "').rstrip(' "')]))
	
data = data.reindex(numpy.random.permutation(data.index)) #don't know why this reindexing is important...

count_vectorizer = CountVectorizer(min_df = 1) #how many times a certain word has to appear in the text to take it into account
counts = count_vectorizer.fit_transform(numpy.asarray(data['text']))

classifier = MultinomialNB()
targets = numpy.asarray(data['class'])
classifier.fit(counts, targets)

#examples = ['linux viagra', "I'm going to attend the Linux users group tomorrow."]
#examples = ['#summer is coming! yay!', "I'm so goddamn angry right now, why is @apple introducing this new feature?"]
examples = csv_test_feed(testing_file, 6, 5, ',')

example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print("Processed predictions: " + str(len(predictions)))
for i in range(len(examples)):
	print(str(predictions[i]) + ": " + examples[i], end="")
#print(predictions)