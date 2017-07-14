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


model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


p = model['hi']
p = torch.from_numpy(p)
p = p.view(1, 300)

zero = torch.Tensor(1,1)
zero.fill_(0)

fineparola = torch.cat([zero, zero], 1)
token = torch.cat([p, fineparola], 1)

checkpoint=torch.load('checkpoint-1.pth')
model_options=checkpoint["model_options"]
model2=Model1(**model_options)


print(model2.dec_to_output)

h = torch.zeros(1, 1, 1024)

input = token
target = token
input = input.unsqueeze(0)

inputlala = Variable(input)#mettendo l'input qui dentro autograd fa si che venga wrappato e reso adatto per l'allenamento
# Forward (use target as decoder input for training)
output, h_nuova = model2(input, Variable(h), Variable(target))



oggetto = model2.forward(inputlala, Variable(h), target_as_input)

print(oggetto)
