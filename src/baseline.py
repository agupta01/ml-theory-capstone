from collections import defaultdict
from collections import Counter
import numpy as np

class BagOfWords:
    
    def __init__(self):
        self.alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 
        'g', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'q', 'r', 
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        
    def vectorize_counts(self, sentence):
        freqs = {i:0 for i in self.alphabet}
        freqs.update(dict(Counter([i for i in sentence])))
        return np.array(list(freqs.values()))

    def bag_of_words(self,fp):
        with open(fp, 'r') as file:
            data = file.readlines()
            
        res = []
        for i in data:
            temp = self.vectorize_counts(self,i)
            res.append(temp)

        bag_of_words = np.array(res).reshape(len(data),26)
        
        return bag_of_words

        
class BigramModel:
    def __init__(self):
        self.bigrams = defaultdict(int)
        self.unigrams = defaultdict(int)
        
    def fit(self, corpus):
        """
        Train the bigram model on the given corpus.
        """
        for sentence in corpus:
            tokens = sentence.strip().split()
            tokens = ['<s>'] + tokens + ['</s>']
            
            for i in range(len(tokens) - 1):
                self.bigrams[(tokens[i], tokens[i + 1])] += 1
                self.unigrams[tokens[i]] += 1
        
    def probability(self, w1, w2):
        """
        Return the probability of the bigram (w1, w2) based on the maximum likelihood estimation.
        """
        if (w1,w2) in self.bigrams and w1 in self.unigrams:
            return self.bigrams[(w1, w2)] / self.unigrams[w1]
        else:
            return "key not found"
        
    def generate(self, w1):
        """
        Generate the next word given the current word w1, using the maximum likelihood estimation.
        """
        if w1 in self.unigrams:
            next_word = max(self.bigrams.keys(),key=lambda x: self.bigrams[(w1, x[1])] if (w1,x[1]) in self.bigrams else 0)
        else:
            return "key not found"
        return next_word


