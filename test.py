import sys
import string
from vector import vector
from gensim.models import Word2Vec
from CornellData import CornellData
import os
import ast
import re
import gensim
import numpy
import torch


print("ChatBot run!")

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)

try:
    while True:
        domanda = input("You: ")
        #Word2Vec

        p = model[domanda]
#        print("vettorizzata: "+ p) #to be removed

#        p = Model1(p)
        print("ChatBot: " +domanda)
except KeyboardInterrupt:
    print('\nBye bye!')
