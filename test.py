import sys
import string
from vector import vector
from gensim.models import Word2Vec
from CornellData import CornellData
import gensim
import numpy as np
import torch
<<<<<<< HEAD

print("ChatBot run!")
#load model of gensim google vector
#model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


model2 = torch.load('checkpoint-2.pth')
<<<<<<< HEAD
token = torch.FloatTensor(1,300)
token.fill_(0)
uno = torch.FloatTensor(1,1)
uno.fill_(1)
zero = torch.FloatTensor(1,1)
zero.fill_(0)
inizio = torch.cat([uno, zero], 1)
fine = torch.cat([zero, uno], 1)
fineparola = torch.cat([zero, zero], 1)

battuta=[]
for parola in model2:
	if (parola==inizio):
		battuta=parola[i+1]
	if parola==fine:
		pass
return battuta


#print(model2)
#load model defined in class copy_example
=======
#insert into for structure
output = model2[:len(model2)-1]
>>>>>>> efee98ef9432dad4c099e7a1ac8f9d4ae8f0a2b1

print(model2)

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
