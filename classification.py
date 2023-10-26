import math
import glob
from ngram import NGram
import os

# Your Code Goes Here

model = NGram()
model.ngram_size = 3
'''
def initiate_model(loaded_vocab:str, model:NGram, ngram_size):

    model.ngram_size = ngram_size
    all_words_str = ''
    for sentence in loaded_vocab:
        all_words_str += sentence
    model.ngrams = model.mle(all_words_str, {})

    return model
'''
#Naive Bayes Calc

#1. Boring Directory Stuff
'''filenames from all pos,neg in each train,test --> 4 lists'''
train_pos_files = []
train_neg_files = []
test_pos_files = []
test_neg_files = []

for filename in os.listdir('sentiment_data/train/pos'):
    train_pos_files.append(filename)


for filename in os.listdir('sentiment_data/train/neg'):
    train_neg_files.append(filename)


for filename in os.listdir('sentiment_data/test/pos'):
    test_pos_files.append(filename)


for filename in os.listdir('sentiment_data/test/neg'):
    test_neg_files.append(filename)

#print(test_pos_files)
#print(test_neg_files)

#2. Training Models
vocab = model.load_vocab('imdb.vocab')

#positive_model
print('training positive model....')
model_train_pos = NGram()
model_train_pos.load_vocab('imdb.vocab')
model_train_pos.train('sentiment_data/train/pos')

#negative_model
print('training negative model....')
model_train_neg = NGram()
model_train_neg.load_vocab('imdb.vocab')
model_train_neg.train('sentiment_data/train/neg')

# 3. Evaluating Test Files

#Positive Model Counts
pos_true_pos = 0
pos_false_pos = 0
pos_true_neg = 0
pos_false_neg = 0
#Negative Model Counts
neg_true_pos = 0
neg_false_pos = 0
neg_true_neg = 0
neg_false_neg = 0

'''Iterate through positive test files and append true positives & false negatives'''
for fname in test_pos_files:
    pos_result = model_train_pos.test('sentiment_data/test/pos/' + fname, 'ppl', {'k':0.014}) #k from part 1
    neg_result = model_train_neg.test('sentiment_data/test/pos/' + fname, 'ppl', {'k':0.014})

    pos_output = (math.log(pos_result)) * (math.log(0.5))
    neg_output = (math.log(neg_result)) * (math.log(0.5))

    print('Postive Model Prediction: ' + str(pos_output))
    print('Negative Model Prediction: ' + str(neg_output))

    if pos_output >= neg_output:
        pos_true_pos += 1
        neg_true_neg += 1
    else:
        pos_false_neg += 1
        neg_false_pos += 1

'''Iterate through negative test files and append false positives & true negatives'''
for fname in test_neg_files:
    pos_result = model_train_pos.test('sentiment_data/test/neg/' + fname, 'ppl', {'k':0.014}) #k from part 1
    neg_result = model_train_neg.test('sentiment_data/test/neg/' + fname, 'ppl', {'k':0.014})

    pos_output = (math.log(pos_result)) * (math.log(0.5))
    neg_output = (math.log(neg_result)) * (math.log(0.5))

    if pos_output > neg_output:
        pos_false_pos += 1
        neg_false_neg += 1
    else:
        pos_true_neg += 1
        neg_true_pos += 1
'''
print('Positive Model Counts:')
print('True Positives: ' + str(pos_true_pos))
print('False Positives: ' + str(pos_false_pos))
print('True Negatives: ' + str(pos_true_neg))
print('False Negatives: ' + str(pos_false_neg))

print('Negative Model Counts:')
print('True Positives: ' + str(pos_true_pos))
print('False Positives: ' + str(pos_false_pos))
print('True Negatives: ' + str(pos_true_neg))
print('False Negatives: ' + str(pos_false_neg))
'''
#Positive Model Metrics
pos_accuracy = (pos_true_pos + pos_true_neg) / (pos_true_pos+pos_true_neg+pos_false_pos+pos_false_neg)
pos_precision = (pos_true_pos) / (pos_true_pos+pos_false_pos)
pos_recall = (pos_true_pos) / (pos_true_pos+pos_false_neg)
pos_f1 = (2*pos_precision*pos_recall) / (pos_precision + pos_recall)

print('Positive Model Metrics')
print('Accuracy: ' + str(pos_accuracy))
print('Precision: ' + str(pos_precision))
print('F1 score: ' + str(pos_f1))
print('************************************************************************')

#Negative Model Metrics
neg_accuracy = (neg_true_pos + neg_true_neg) / (neg_true_pos+neg_true_neg+neg_false_pos+neg_false_neg)
neg_precision = (neg_true_pos) / (neg_true_pos+neg_false_pos)
neg_recall = (neg_true_pos) / (neg_true_pos+neg_false_neg)
neg_f1 = (2*neg_precision*neg_recall) / (neg_precision + neg_recall)

print('Negative Model Metrics')
print('Accuracy: ' + str(neg_accuracy))
print('Precision: ' + str(neg_precision))
print('F1 score: ' + str(neg_f1))
