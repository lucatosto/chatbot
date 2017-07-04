import sys
import string
from vector import vector
from gensim.models import Word2Vec
from CornellData import CornellData

print("ChatBot run!")

try:
    while True:
        domanda = input("You: ")
        #Word2Vec
        p = model[domanda]
        print("vettorizzata: "+ p) #to be removed
        p = Model1(p)
        print("ChatBot: " +p)
except KeyboardInterrupt:
    print('\nBye bye!')
