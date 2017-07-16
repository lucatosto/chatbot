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

class Options():
    pass
opt = Options()

# Model options
opt.encoder_layers = 1
opt.lstm_size = 1024

# Test options
opt.model = sys.argv[1] if len(sys.argv) > 1 else None
opt.test = sys.argv[2] if len(sys.argv) > 2 else None

# Backend options
opt.no_cuda = True


#load gensim model
#model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

#load my model
checkpoint=torch.load('checkpoint-1.pth')
model_options=checkpoint["model_options"]

model2=Model1(**model_options)
model2.load_state_dict(checkpoint["model_state"])

#prepare to test
zero = torch.FloatTensor(1,1)
zero.fill_(0)
fineparola = torch.cat([zero, zero], 1)

h=torch.zeros(1,1,1024)
target_as_input=torch.zeros(1,302)

try:
    domanda=input("you: ")
    domanda = re.findall(r'\w+', domanda)
    vettoreparole=torch.FloatTensor()

    for parola in domanda:
        try:
            p = model[parola]
            p = torch.from_numpy(p)
            p = p.view(1, 300)
            p = torch.cat([p, fineparola],1)
        except:
            pass
        vettoreparole = torch.cat([vettoreparole, p])

    print (vettoreparole)
    print("FINE VETTORE PAROLE")

    vettoreparole=vettoreparole.unsqueeze(0)
    target_as_input=target_as_input.unsqueeze(0)

    output = model2(Variable(vettoreparole), Variable(h), Variable(target_as_input))
    print(output)
    print("fine output")

    vettoreparole2=[]
    for parola in output:
        print(parola)
        p = (parola) #da torch.FloatTensor ad array
        uscita = model.similar_by_vector([p, p], topn=1)
        vettoreparole2.append(uscita)
    #print(vettoreparole2)
    print("fine vettoreparole2")

except KeyboardInterrupt:
	print("Bye")
