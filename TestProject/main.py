from __future__ import unicode_literals

import glob
import multiprocessing
import os

import numpy as np
import pandas as pd
from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from hazm import *

all_rec = glob.iglob(os.path.join('./data', "*.txt"), recursive=True)
dataframes = (
    pd.read_csv(f, sep="\t", encoding="utf8", header=None, names=["ID", "Text"], dtype={'ID': np.str, 'Text': np.str})
    for f
    in all_rec)
df = pd.concat(dataframes, ignore_index=True)
text_corpus = df['Text'][:70000]


def date_cleaner():
    sentences = []
    normalizer = Normalizer()
    stemmer = Stemmer()
    stops = set(stopwords_list())
    for i in range(len(text_corpus)):
        text_corpus[i] = normalizer.normalize(text_corpus[i])
        words = [[stemmer.stem(word) for word in word_tokenize(sentence) if word not in stops] for sentence in
                 sent_tokenize(text_corpus[i])]
        sentences.append(words[0])
    return sentences


sent = date_cleaner()
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]
num_cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=num_cores - 1)
w2v_model.build_vocab(sentences, progress_per=10000)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.init_sims(replace=True)
print(w2v_model.wv.most_similar(positive=["زمستون"]))