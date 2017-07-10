class Options():
    pass
opt = Options()
# Training options
opt.batch_size = 64
opt.learning_rate = 0.001
opt.learning_rate_decay_by = 0.8
opt.learning_rate_decay_every = 10
opt.weight_decay = 5e-4
opt.momentum = 0.9
opt.data_workers = 0
opt.epochs = 10
# Checkpoint options
opt.save_every = 2
# Model options
opt.encoder_layers = 1
opt.lstm_size = 1024
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
a = vector()
dataset_totale= a.vettorizzazione()
train_dataset=dataset_totale[:20]
test_dataset=dataset_totale[21:25]


print("ChatBot run!")
model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
#model2 = torch.load('checkpoint-2.pth')
token = torch.FloatTensor(1,300)
token.fill_(0)
uno = torch.FloatTensor(1,1)
uno.fill_(1)
zero = torch.FloatTensor(1,1)
zero.fill_(0)
inizio = torch.cat([uno, zero], 1)
fine = torch.cat([zero, uno], 1)
fineparola = torch.cat([zero, zero], 1)

sos_idx = torch.cat([token, inizio], 1)
eos_idx = torch.cat([token, fine], 1)
#model_options = {'input_size': 302, 'sos_idx': sos_idx, 'eos_idx': eos_idx, 'encoder_layers': opt.encoder_layers, 'lstm_size': opt.lstm_size}
model_options = {'input_size': train_dataset[0][0].size(1), 'sos_idx': sos_idx, 'eos_idx': eos_idx, 'encoder_layers': opt.encoder_layers, 'lstm_size': opt.lstm_size}
model2 = Model1(**model_options)
optimizer = torch.optim.SGD(model.parameters(), lr = opt.learning_rate, momentum = opt.momentum, weight_decay = opt.weight_decay)
model2.train()
model2.load_state_dict()
try:
    while True:
        domanda = input("You: ")

        domandavettorizzata = model[domanda]

        rispostavettorizzata = np.array( domandavettorizzata , dtype='f')
        #risposta = model2.most_similar( [ rispostavettorizzata ], [], 1)

        print("ChatBot: " +risposta[0])
except KeyboardInterrupt:
    print('\nBye bye!')
