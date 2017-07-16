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
# Test options
import sys
opt.model = sys.argv[1] if len(sys.argv) > 1 else None
opt.test = sys.argv[2] if len(sys.argv) > 2 else None
# Backend options
opt.no_cuda = True
import string
import re
import sys
import os
import time
from vector import vector
import gensim
from gensim.models import Word2Vec
from CornellData import CornellData
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from Model1 import Model1

#dopo aver richiamato la classe per il model faccio la load

#load gensim model
model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

#load my model
checkpoint=torch.load('checkpoint-2.pth')
model_options=checkpoint["model_options"]
model2=Model1(**model_options)
#hidden=checkpoint["h"]
model2.load_state_dict(checkpoint["model_state"])


#modello_h=modello.h
#modello_dict=modello.x
#prepare to test
zero = torch.FloatTensor(1,1)
zero.fill_(0)
fineparola = torch.cat([zero, zero], 1)
h=torch.zeros(1,1,1024)
target_as_input=torch.zeros(1,302)
#print(fineparola)
try:
    domanda=input("you: ")
    domanda = re.findall(r'\w+', domanda)
    vettoreparole=torch.FloatTensor()
    for parola in domanda:
        #print(parola)
        try:
            p = model[parola]
            p = torch.from_numpy(p)
            p = p.view(1, 300)
            p = torch.cat([p, fineparola],1)
        except:
            pass
    #retro=output2.h or modello_h
    #output = modello.forward(modello_dict, retro, vettoreparole)  
    #print(output.x)
        vettoreparole = torch.cat([vettoreparole, p])
    print (vettoreparole)
    vettoreparole=vettoreparole.unsqueeze(0)
    target_as_input=target_as_input.unsqueeze(0)
    output, h_nuova=model2(Variable(vettoreparole), Variable(h), Variable(target_as_input))
    print(output)
    vettoreparole2=[]
    for parola in output:
        uscita=model.most_similar(positive= parola, topn=1)
        vettoreparole2.append(uscita)
    print(vettoreparole2)
    #output=model2(Variable(model2.load_state_dict(checkpoint["model_state"]), Variable(h), vettoreparole))
    #risposta=Variable(risposta)
    #h= Variable(torch.zeros(1, 64, 1024))
    #retro=output2.h or modello_h
    #output2=modello.forward(modello_dict,output.h, vettoreparole)
    #risposta=model2(vettoreparole)
    #print(risposta)
except KeyboardInterrupt:
	print("Bye")
