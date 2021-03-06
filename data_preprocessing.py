import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import pandas as pd
import random
import pickle
from collections import Counter
lemmatizer = WordNetLemmatizer()

def init_process(fin,fout):
    outfile = open(fout,'w',encoding='latin-1')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0]
                elif initial_polarity == '4':
                    initial_polarity = [0,1]

                tweet = line.split(',')[-1]
                outline = str(initial_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()


def create_lexicon(fin):
	lexicon = []
	with open(fin,'r',buffering = 100000,encoding='latin-1') as f:
		try:
			content = ''
			counter = 1
			for line in f:
				counter += 1
				if (counter/2500.0).is_integer():
					tweet = line.split(':::')[1]
					content += ''+tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon + words))
					print(counter, len(lexicon))

		except Exception as e:
			print(str(e))
	with open('lexicon.pickle','wb') as f:
		pickle.dump(lexicon,f)






def convert_to_vec(fin,fout,lexicon_pickle):
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)

    outfile = open(fout,'w',encoding='latin-1')
    counter = 0
    with open(fin,'r',buffering=200000,encoding='latin-1') as f:
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            words = word_tokenize(line.lower())
            words = [lemmatizer.lemmatize(i) for i in words]

            features = np.zeros(len(lexicon))

            for w in words:
                if w.lower() in lexicon:
                    index_value = lexicon.index(w.lower())
                    features[index_value]+=1

            feature_sets = list(features)
            outline = str(feature_sets)+':::'+label+'\n'
            outfile.write(outline)

    print(counter)
    outfile.close()

def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False,encoding='latin-1')
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('train_set_shuffled.csv', index=False)
    

def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=200000,encoding='latin-1') as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)


if __name__ == '__main__':
    init_process('training.1600000.processed.noemoticon.csv','train_set.csv')
    init_process('testdata.manual.2009.06.14.csv','test_set.csv')
    create_lexicon('train_set.csv')
    convert_to_vec('test_set.csv','processed-test-set.csv','lexicon.pickle')
    shuffle_data('train_set.csv')
    create_test_data_pickle('processed-test-set.csv')



