import os
import torch
import json
import re
from tqdm import tqdm
import random

filter_symbols = re.compile('[a-zA-Z]*')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            i = len(self.idx2word)
            self.word2idx[word] = i
            self.idx2word.append(word)
        else:
            raise ValueError("Please don't call this method, so we won't break the dictionary :) ")

    def __len__(self):
        return len(self.idx2word)

def get_sentence(sentence, dictionary):
    return [dictionary.idx2word[idx] for idx in sentence]

def get_word_list(line, dictionary):
    splitted_words = json.loads(line.lower()).split()
    words = ['<bos>']
    for word in splitted_words:
        word = filter_symbols.search(word)[0]
        if len(word)>1:
            if dictionary.word2idx.get(word, False):
                words.append(word)
            else:
                words.append('<unk>')
    words.append('<eos>')

    return words


class Corpus(object):
    def __init__(self, dictionary, input_tweets, aa_test_tweets, wh_test_tweets):
        self.dictionary = dictionary
        self.train = self.tokenize(input_tweets)
        self.aa_test_tweets = self.tokenize(aa_test_tweets)
        self.wh_test_tweets = self.tokenize(wh_test_tweets)

    def tokenize(self, input_tweets):
        """Tokenizes a text file."""
        tokens = 0
        total_sen = 0
        random.shuffle(input_tweets)

        for words in input_tweets:
            tokens += len(words)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for words in input_tweets:
            for word in words:
                ids[token] = word
                token += 1

        return ids