import sys
import string
from vector import vector
from gensim.models import Word2Vec
from CornellData import CornellData
import gensim
import numpy as np
import torch

print("ChatBot run!")
#load model of gensim google vector
model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


model2 = torch.load('checkpoint-2.pth')

battuta=[]
for parola in model2:
	battuta=parola[0:300]



#print(model2)
#load model defined in class copy_example
#insert into for structure


print(battuta)

output = model.most_similar(positive = 'hello', topn = 1)

output = output.data()
output = output.numpy

print(output)
#load model defined in class copy_example

#model_options = {'input_size': train_dataset[0][0].size(1), 'sos_idx': sos_idx, 'eos_idx': eos_idx, 'encoder_layers': opt.encoder_layers, 'lstm_size': opt.lstm_size}



try:
    while True:
        domanda = input("You: ")
        #pseudo code
        #domandavettorizzata = model[domanda]

        domandavettorizzata = model[domanda]



        rispostavettorizzata = np.array( domandavettorizzata , dtype='f')
        risposta = model2( [ rispostavettorizzata ], [], 1)
        gensim.most_similar(risposta)

        print("ChatBot: " +domanda)
except KeyboardInterrupt:
    print('\nBye bye!')
