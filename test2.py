import sys
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import re
import gensim
import numpy
#model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

zero = torch.FloatTensor(1,1)
zero.fill_(0)
fineparola = torch.cat([zero, zero], 1)

domanda=input("you: ")
domanda = re.findall(r'\w+', domanda)
vettoreparole=torch.FloatTensor()
for parola in domanda:
    print(parola)
    try:
        p = model[parola]
        p = torch.from_numpy(p)
        p = p.view(1, 300)
        p = torch.cat([p, fineparola],1)
    except:
        pass
    vettoreparole = torch.cat([vettoreparole, p])
print (vettoreparole)