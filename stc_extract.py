import argparse, os, random, re

POSITIVE = '4'
NEGATIVE = '0'

parser = argparse.ArgumentParser(description='Create subset of the stanford twitter corpus.')
parser.add_argument('output_filename', action='store', help='Name of the output file, will be saved as a .csv file, placed in /data/ folder!')
parser.add_argument('--lines', action='store', type=int, default=100, help='Number of lines to extract!')
args = parser.parse_args()

path = 'data/training.1600000.processed.noemoticon.csv'
size = os.path.getsize(path)
lines = args.lines

def decently_distributed(positives, negative):
	if abs(len(positives) / float(lines) - len(negatives) / float(lines)) <= 0.05:
		return True
	else:
		return False

def pop_n(list, n):
	for i in xrange(n):
		list.pop()

positives = []
negatives = []
processed_lines = set()
with open(path) as file:
	while len(positives) + len(negatives) < lines:
		# grab random lines from file, store in separate lists based on polarity
		while len(positives) + len(negatives) < lines:
			file.seek(random.randrange(size), 0)
			file.readline() # file.seek most likely puts the file object's position in the middle of a line, so jump to the next line
			position_of_line = file.tell()
			line = file.readline().rstrip('\n')
			if line != '' and position_of_line not in processed_lines:
				polarity = line[1] # polarity is given by the second character of each line
				processed_lines.add(position_of_line)
				if polarity == POSITIVE:
					positives.append(line)
				elif polarity == NEGATIVE:
					negatives.append(line)
		# remove some lines if positives and negatives arn't evenly enough sized, then repeat random grabbing loop
		if not decently_distributed(positives, negatives):
			larger_list = []
			if len(positives) > len(negatives):
				larger_list = positives
			else:
				larger_list = negatives
			pop_n(larger_list, abs(len(positives) - len(negatives)))
	
output_filename = 'data/' + re.sub(r'\.csv$', '', args.output_filename) + '.csv'
with open(output_filename, 'w+') as file:
	for tweet in positives:
		file.write(tweet + '\n')
	for tweet in negatives:
		file.write(tweet + '\n')

print(" - File saved as: " + output_filename)
print(" - Positive tweets: " + str(len(positives)))
print(" - Negative tweets: " + str(len(negatives)))
