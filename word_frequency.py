import nltk
from nltk.corpus import webtext
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer

nltk.download('webtext')
val = nltk.data.load('val.txt')

tokenizer = RegexpTokenizer(r'\w+')
wt_words = tokenizer.tokenize(val)
data_analysis = nltk.FreqDist(wt_words)

# Let's take the specific words only if their frequency is greater than 3.
filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 3])
for key in sorted(filter_words):
    print("%s: %s" % (key, filter_words[key]))
data_analysis = nltk.FreqDist(filter_words)
data_analysis.plot(25, cumulative=False)
