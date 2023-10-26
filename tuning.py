'''I iterate through each decimal place or each two decimal places (10-100 iterations)
to find the local minimum for Ks and Lambdas. I tune for the first Lambda(L1) and set the other
two lambdas to be (1-dec)/2. Currently, all of the other for loops are commented out, and 
only the 25-epoch windows corresponding to the local minimum on the training set is shown.

I use the optimized K and L1 25-epoch windows from the training data on the test and eval data to
find their optimized values.

Feel free to remove the comments on the other loops, but the code will run for a while :) 
'''

from ngram import NGram

model = NGram()
model.ngram_size = 3

#Train Data
loaded_vocab = model.load_vocab('train_data/train.txt')
model.train('train_data')
#Optimal Lamdas: [0.061, 0.4695, 0.4695]
#Optimal K: 0.014
'''
Optimal K: 0.014
Lowest Perplexity (K): 510.42334003336157

Optimal L1: 0.061
Optimal L2: 0.46950000000000003
Optimal L3: 0.46950000000000003
Lowest Interpolation (Lambdas): 181.88245101667073
'''

#Test_Data
'''
loaded_vocab = model.load_vocab('test_data/test.txt')
model.train('test_data')

Optimal K: 0.001
Lowest Perplexity (K): 26.991365020972243
Optimal L1: 0.097
Optimal L2: 0.4515
Optimal L3: 0.4515
Lowest Interpolation (Lambdas): 29.779129723685845
'''

#Eval_Data
'''
loaded_vocab = model.load_vocab('eval_data/eval.txt')
model.train('eval_data')

Optimal K: 0.015
Lowest Perplexity (K): 698.9074098877103
Optimal L1: 0.051000000000000004
Optimal L2: 0.4745
Optimal L3: 0.4745
Lowest Interpolation (Lambdas): 281.01807869941206
'''


all_words_str = ''
for sentence in loaded_vocab:
    all_words_str += sentence
three_grams = model.get_ngrams(all_words_str, 3)
model.ngrams = model.mle(all_words_str, {})

#optimal K: whole numbers
'''
k_whole_results = {}
for i in range(10):
    pred = model.test('test_data/test.txt', 'ppl', {'k':int(i)})
    k_whole_results[i] = pred
    print('epoch (K whole numbers) ' + str(i) + ':')
    print(pred)
'''

optimal_k = 1

#Optimal K to the nearest 2 decimals
'''
k_2dec_results = {}
for i in range(10):
    dec = i * 0.01
    pred = model.test('test_data/test.txt', 'ppl', {'k':float(dec)})
    k_2dec_results[dec] = pred
    print('epoch (K 2decimal numbers) ' + str(i) + ':')
    print(pred)
'''

optimal_k = 0.01

#Optimal K to the nearest 4 decimals (this one runs for a full 100 epochs, keep in docstring!)
'''
k_4dec_results = {}
for i in range(100):
    dec = i * 0.0001
    pred = model.test('test_data/test.txt', 'ppl', {'k':float(dec)})
    k_4dec_results[dec] = pred
    print('epoch (K decimal numbers) ' + str(i) + ':')
    print(pred)

#from 4 decimals
optimal_k = 0.0099
optimal_perplexity = 513.7950953436864
'''
#Optimal K to the nearest 3 decimals, final loop
k_3dec_results = {}
perplexity_values = []
for i in range(25):
    dec = i * 0.001
    pred = model.test('test_data/test.txt', 'ppl', {'k':float(dec)})
    k_3dec_results[dec] = pred
    perplexity_values.append(pred)
    print('epoch (K 3decimal numbers) ' + str(i) + ':')
    print(pred)

optimal_perplexity = min(perplexity_values)
optimal_k = list(k_3dec_results.keys())[list(k_3dec_results.values()).index(optimal_perplexity)]

print('Optimal K: ' + str(optimal_k))
print('Lowest Perplexity (K): ' + str(optimal_perplexity))


# Optimal K: 0.014
# Lowest Perplexity (K): 510.0095816956241

#optimal Lambda: 1dec 
'''
L_1dec_results = {}
for i in range(10):
    dec = i*0.1
    pred = model.test('test_data/test.txt', 'ppl', {'lambdas':[dec, ((1-dec)/2), ((1-dec)/2)]})
    L_1dec_results[i] = pred
    print('epoch (Lambda 1dec) ' + str(i) + ':')
    print('L1: ' + str(dec) + ' L2: ' + str(((1-dec)/2)) + " L3: " + str(((1-dec)/2)))
    print('interpolation: ' + str(pred))
'''

optimal_lambda = 0.1

#optimal Lambda: 3dec, this one runs for full epochs or until assertion error, keep in docstring!
'''
L_3dec_results = {}
for i in range(100):
    dec = i*0.001
    pred = model.test('test_data/test.txt', 'ppl', {'lambdas':[dec, ((1-dec)/2), ((1-dec)/2)]})
    L_3dec_results[i] = pred
    print('epoch (Lambda 1dec) ' + str(i) + ':')
    print('L1: ' + str(dec) + ' L2: ' + str(((1-dec)/2)) + " L3: " + str(((1-dec)/2)))
    print('interpolation: ' + str(pred))
'''
optimal_lambda = 0.061

#lambda 2 decimal results: this runs for 25 total epochs, final loop
L_2dec_results = {}
interpolation_values = []
for i in range(51, 99, 2):
    dec = i*0.001
    pred = model.test('test_data/test.txt', 'ppl', {'lambdas':[dec, ((1-dec)/2), ((1-dec)/2)]})
    L_2dec_results[dec] = pred
    interpolation_values.append(pred)
    print('epoch (Lambda 2dec) ' + str(i) + ':')
    print('L1: ' + str(dec) + ' L2: ' + str(((1-dec)/2)) + " L3: " + str(((1-dec)/2)))
    print('interpolation: ' + str(pred))

optimal_interpolation = min(interpolation_values)
optimal_L = list(L_2dec_results.keys())[list(L_2dec_results.values()).index(optimal_interpolation)]

print('Optimal L1: ' + str(optimal_L))
print('Optimal L2: ' + str(((1-optimal_L)/2)))
print('Optimal L3: ' + str(((1-optimal_L)/2)))
print('Lowest Interpolation (Lambdas): ' + str(optimal_interpolation))

print('Optimal K: ' + str(optimal_k))
print('Lowest Perplexity (K): ' + str(optimal_perplexity))









