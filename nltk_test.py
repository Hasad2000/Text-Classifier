from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = 'I am a sucker for medieval castles :) & turtles'
tokens = word_tokenize(text)
print(tokens)

#stop_words = set(stopwords.words('english'))
#print(stop_words)