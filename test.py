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
#model2 = TODO import our pth model

try:
    while True:
        domanda = input("You: ")

        domandavettorizzata = model[domanda]

        rispostavettorizzata = np.array( domandavettorizzata , dtype='f')
        risposta = model.most_similar( [ rispostavettorizzata ], [], 1)

        print("ChatBot: " +risposta)
except KeyboardInterrupt:
    print('\nBye bye!')
