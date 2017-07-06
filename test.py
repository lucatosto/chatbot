import sys
import string
from vector import vector
from gensim.models import Word2Vec
from CornellData import CornellData
import os
import ast
import re
import gensim
import numpy as np
import torch


print("ChatBot run!")

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)

try:
    while True:
        domanda = input("You: ")
        #Word2Vec
        #p = model[p]
        risposta = model[domanda]  #Questo model sarà Model1 quello allenato. e 'domanda' sarà 'p'
        model_word_vector = np.array( risposta , dtype='f')
        most_similar_words = model.most_similar( [ model_word_vector ], [], 1)
#        print("vettorizzata: "+ p) #to be removed

        print("ChatBot: " +most_similar_words)
except KeyboardInterrupt:
    print('\nBye bye!')
