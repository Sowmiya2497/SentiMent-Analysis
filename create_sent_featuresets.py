import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000

def create_lexicon(pos,neg):
	lexicon = []
	for fi in [pos,neg]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for line in contents[:hm_lines]:
				words = word_tokenize(line.lower())
				lexicon += list(words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2=[]
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)

	print(len(l2))
	return l2

def sample_handling(sample,lexicon,classification):
	featuresets = [ ]
	with open(sample,'r') as f:
		lines = f.readlines()
		for l in lines[:hm_lines]:
			features = np.zeros(len(lexicon))
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			for w in current_words:
				if w.lower() in lexicon:
					index_value = lexicon.index(w.lower())
					features[index_value] += 1
			features = list(features)
			featuresets.append([features,classification])

	return featuresets


def create_feature_sets_and_labels(pos,neg,test_size=0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling(pos,lexicon,[1,0])
	features += sample_handling(neg,lexicon,[0,1])
	random.shuffle(features)

	testing_size = int(test_size*len(features))
	features = np.array(features)
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,0][-testing_size:])

	return train_x,train_y,test_x,test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)




