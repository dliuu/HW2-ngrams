import math
import glob
from ngram import NGram
import os

# Your Code Goes Here
model = NGram()
model.ngram_size = 3

def initiate_model(loaded_vocab:str, model:NGram, ngram_size):

    model.ngram_size = ngram_size
    all_words_str = ''
    for sentence in loaded_vocab:
        all_words_str += sentence
    model.ngrams = model.mle(all_words_str, {})

    return model

#Training positive_model
'''
vocab = model.load_vocab('imdb.vocab')
model_train_pos = initiate_model(vocab, model, 1)
model_train_pos = model.train('sentiment_data/train/pos')
'''

#Naive Bayes Calc

#boring directory stuff because I forgot about glob
return_list = []
directory = 'sentiment_data'

for directory_name in os.listdir(directory):
    directory_name = os.path.join(directory, directory_name)
    
    for pos_neg_directory in os.listdir(directory_name):
        pos_neg = os.path.join(directory_name, pos_neg_directory)
    
        for filename in os.listdir(pos_neg):
            return_list.append(filename)

#Training negative_model
'''
model_train_neg = initiate_model(vocab, model, 1)
model_train_neg = model.train('sentiment_data/train/neg')
'''




