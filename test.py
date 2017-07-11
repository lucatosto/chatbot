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
#import copy_example

print("ChatBot run!")
#load model of gensim google vector
model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
model2 = torch.load('checkpoint-2.pth')
#print(model2)
#load model defined in class copy_example

#torch.load({'model_options': model_options, 'model_state': model.state_dict()}, './checkpoint-2.pth')

#torch.save({'model_options': model_options, 'model_state': model.state_dict()}, checkpoint_path)

#model_options = {'input_size': train_dataset[0][0].size(1), 'sos_idx': sos_idx, 'eos_idx': eos_idx, 'encoder_layers': opt.encoder_layers, 'lstm_size': opt.lstm_size}



try:
    while True:
        domanda = input("You: ")

        domandavettorizzata = model[domanda]

        rispostavettorizzata = np.array( domandavettorizzata , dtype='f')
        risposta = model2( [ rispostavettorizzata ], [], 1)
        gensim.most_similar(risposta)

        print("ChatBot: " +risposta[0])
except KeyboardInterrupt:
    print('\nBye bye!')
