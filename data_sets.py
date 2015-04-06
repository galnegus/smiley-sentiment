import csv, feature_reduction

def _stanford_parse(filename):
	data = {'text': [], 'class': []}
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile, fieldnames=('polarity', 'id', 'date', 'query', 'user', 'tweet'))
		for row in reader:
			if row['polarity'] in ('0', '4'):
				data['text'].append(feature_reduction.reduce(row['tweet']))
				data['class'].append(row['polarity'])
	return data

def _sts_gold_parse(filename):
	data = {'text': [], 'class': []}
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile, delimiter=';')
		for row in reader:
			if row['polarity'] in ('0', '4'):
				data['text'].append(feature_reduction.reduce(row['tweet']))
				data['class'].append(row['polarity'])
	return data

# testing data
def testing():
	return _stanford_parse('data/testing.csv')

# hand annotated training data
def training_sts_gold():
	return _sts_gold_parse('data/training.sts_gold.csv')

# automatically (smiley) annotated training data
def training_2000():
	return _stanford_parse('data/training.2000.csv')
def training_10000():
	return _stanford_parse('data/training.10000.csv')
def training_100000():
	return _stanford_parse('data/training.100000.csv')
def training_1600000():
	return _stanford_parse('data/training.1600000.processed.noemoticon.csv')
