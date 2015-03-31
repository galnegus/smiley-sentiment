import re

user_pattern = re.compile(r'(?:^|(?<=[^@\w]))(@\w{1,15})\b')
user_token = 'USER'

url_pattern = re.compile(r'(?:https?|ftp)://[^\s/$.?#].[^\s]*')
url_token = 'URL'

repeating_pattern = re.compile(r'(.)\1{2,}')
def repeating_token(matchobj):
	return matchobj.group(1) * 2

def reduce(tweet):
	tweet = user_pattern.sub(user_token, tweet)
	tweet = url_pattern.sub(url_token, tweet)
	tweet = repeating_pattern.sub(repeating_token, tweet)

	return tweet
