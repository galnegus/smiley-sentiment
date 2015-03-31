import re, unittest
from feature_reduction import *

class UnitTests(unittest.TestCase):

	# user pattern tests
	def test_user_pattern_token(self):
		replace_test = user_pattern.sub(user_token, '@lololol, h@h@h@ ipsum@dolor.co \'@aaa\'')
		replace_answer = user_token + ', h@h@h@ ipsum@dolor.co \'' + user_token + '\''
		self.assertEqual(replace_test, replace_answer)

	def test_user_pattern_match(self):
		self.assertTrue(user_pattern.fullmatch('@abc'))
		self.assertTrue(user_pattern.fullmatch('@a2____224'))
		self.assertTrue(user_pattern.fullmatch('@108'))

	def test_user_pattern_fail(self):
		self.assertFalse(user_pattern.fullmatch('fuu@bar'))
		self.assertFalse(user_pattern.fullmatch('@abc.de'))
		self.assertFalse(user_pattern.fullmatch('\'@kek'))
		self.assertFalse(user_pattern.fullmatch('@1234567890123456'))
		self.assertFalse(user_pattern.fullmatch(''))

	# url pattern tests
	def test_url_pattern_token(self):
		replace_test = url_pattern.sub(url_token, 'http://foo.com/blah_blah test https://142.42.1.1/')
		replace_answer = url_token + ' test ' + url_token
		self.assertEqual(replace_test, replace_answer)

	def test_url_pattern_match(self):
		self.assertTrue(url_pattern.fullmatch("http://www.com"))
		self.assertTrue(url_pattern.fullmatch("https://142.42.1.1/"))
		self.assertTrue(url_pattern.fullmatch("https://www.example.com/foo/?bar=baz&inga=42&quux"))

	def test_url_pattern_fail(self):
		self.assertFalse(url_pattern.fullmatch('http:/'))
		self.assertFalse(url_pattern.fullmatch('abc'))
		self.assertFalse(url_pattern.fullmatch('http://www .website .com'))
		self.assertFalse(url_pattern.fullmatch(''))

	# more url tests (if needed)
	# https://mathiasbynens.be/demo/url-regex

	# repeating pattern tests
	def test_repeating_pattern(self):
		replace_test = repeating_pattern.sub(repeating_token, 'yyyyeeeeeeaaaaaaahhhhhhhh....... fooootball')
		replace_answer = 'yyeeaahh.. football'
		self.assertEqual(replace_test, replace_answer)

	def test_repeating_match(self):
		self.assertTrue(repeating_pattern.fullmatch('ooooooooo'))
		self.assertTrue(repeating_pattern.fullmatch('......'))
		self.assertTrue(repeating_pattern.fullmatch('111'))

	def test_repeating_fail(self):
		self.assertFalse(repeating_pattern.fullmatch('oo'))
		self.assertFalse(repeating_pattern.fullmatch('11112222'))
		self.assertFalse(repeating_pattern.fullmatch(''))

	# reduce tests
	def test_reduce(self):
		test_1 = reduce('@angelicbiscuit http://twitpic.com/7pf62 - not a &quot;get lost in  melbourne&quot; ad rip off  we r sydney :p')
		answer_1 = user_token + ' ' + url_token + ' - not a &quot;get lost in  melbourne&quot; ad rip off  we r sydney :p'
		self.assertEqual(test_1, answer_1)

		test_2 = reduce('im soooooooooooooo booooooooooooooooooorrrrrrrrrrrrred an i havent even been finished exams for a day  damn you xbox live. damn u to hell')
		answer_2 = 'im soo boorred an i havent even been finished exams for a day  damn you xbox live. damn u to hell'
		self.assertEqual(test_2, answer_2)

		test_3 = reduce('i\'ve only been in sydney for 3 hrs but I miss my friends  especially @ktjade!!!')
		answer_3 = 'i\'ve only been in sydney for 3 hrs but I miss my friends  especially ' + user_token + '!!'
		self.assertEqual(test_3, answer_3)


if __name__ == '__main__':
    unittest.main()
