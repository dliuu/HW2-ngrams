import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngram import NGram 

def testByWordMetrics():
        with open('tiny_data/cat.txt', 'r') as f:
            data = f.read()
            model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
            model.ngrams = model.mle(data, {})
            model.byWordMetrics('the dog jumps over the cat', {'k': 1})

testByWordMetrics()