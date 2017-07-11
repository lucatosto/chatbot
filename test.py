import string
from vector import vector
from gensim.models import Word2Vec
from CornellData import CornellData
import gensim
import numpy as np
import torch
import re
import sys
import os
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
class Model1(nn.Module):

    def __init__(self, input_size, sos_idx, eos_idx, encoder_layers = 1, lstm_size = 128):
        # Call parent
        super(Model1, self).__init__()
        # Set attributes
        self.is_cuda = False
        self.input_size = input_size
        self.encoder_layers = encoder_layers
        self.lstm_size = lstm_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        # Define modules
        self.encoder = nn.LSTM(input_size = self.input_size, hidden_size = self.lstm_size, num_layers = encoder_layers, batch_first = True, dropout = 0, bidirectional = False)
        self.decoder = nn.LSTM(input_size = self.input_size, hidden_size = self.lstm_size, num_layers = 1,              batch_first = True, dropout = 0, bidirectional = False)
        self.dec_to_output = nn.Linear(self.lstm_size, self.input_size)

    def cuda(self):
        self.is_cuda = True
        super(Model1, self).cuda()

    def forward(self, x, h, target_as_input):
        # Get input info
        batch_size = x.size(0)  #100
        seq_len = x.size(1)
        # Initial state
        c_0 = Variable(torch.zeros(self.encoder_layers, batch_size, self.lstm_size))#c_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch
        # Check CUDA
        if self.is_cuda:
            c_0 = c_0.cuda(async = True)
        # Compute encoder output (hidden layer at last time step)
        x = self.encoder(x, (h, c_0))[0][:,-1,:]
        # Prepare decoder state
        h_0 = x.unsqueeze(0) # Adds num_layers dimension
        c_0 = Variable(torch.zeros(h_0.size()))

        # Check CUDA
        if self.is_cuda:
            c_0 = c_0.cuda(async = True)
        # If target_as_input is provided (during training), target is input sequence; otherwise output is fed back as input
        if target_as_input is not None:
            # Compute decoder output
            x = self.decoder(target_as_input, (h_0, c_0))[0].contiguous()
            # Compute outputs
            #print(x.data.size())
            x = x.view(-1, self.lstm_size)
            #print(x.data.size())
            x = self.dec_to_output(x)
            #print(x.data.size())
            # Compute softmax
            x = F.log_softmax(x)
            #print(x.data.size())
            x = x.view(batch_size, target_as_input.size(1), -1)
        else:
            # Initialize input
            input = torch.zeros(batch_size, 1, self.input_size)
            input[:, :, self.sos_idx].fill_(1)
            input = Variable(input)
            if self.is_cuda:
                input = input.cuda()
            h = h_0
            c = c_0
            # Initialize list of outputs at each time step
            output = []
            # Process until EOS is found or limit is reached
            for i in range(0, 200): #this must be dependent on dataset
                # Get decoder output at this time step
                o, hc = self.decoder(input, (h, c))
                h, c = hc
                # Compute output
                o2 = self.dec_to_output(o.view(-1, self.lstm_size))
                # Compute log-softmax
                o2 = F.log_softmax(o2)
                # View as sequence and add to outputs
                o2 = o2.view(batch_size, 1, -1)
                output.append(o2)
                # Compute predicted outputs
                output_idx = o2.data.max(2)[1].squeeze()
                # Check all words are in EOS
                if (output_idx == self.eos_idx).all():
                    break
                # Compute input for next step    l'uscita del primo hidden layer
                input = torch.zeros(batch_size, 1, self.input_size)
                for j in range(0, batch_size):
                    input[j, 0, output_idx[j]] = 1
                input = Variable(input)
                if self.is_cuda:
                    input = input.cuda(async = True)
            # Concatenate all log-softmax outputs
            x = torch.cat(output, 1)
        return x, h_0

model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
checkpoint = torch.load('checkpoint-2.pth')
model_options = checkpoint["model_options"]
model2 = Model1(**model_options)
model2 = model2.load_state_dict(checkpoint["model_state"])
zero = torch.FloatTensor(1,1)
zero.fill_(0)
fineparola = torch.cat([zero, zero], 1)
try:
    stato=[]
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
    #pseudo-codice. Non sapendo cosa torna non sappiamo come concatenarlo
    stato2 = stato + vettoreparole
    risposta=model2(stato2)
    print(risposta)
    vettoreparole2=[]
    for parola in risposta:
        output = model.most_similar(positive = parola, topn = 1)
        vettoreparole2.append(output)
except KeyboardInterrupt:
    print('\nBye bye!')


"""

#load model of gensim google vector
model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

   #e' giusto?
#model2[0] SOS   ---- Manca ciclo
parola = model2[1]
parola = parola[0:300]

output = model.most_similar(positive = parola, topn = 1)
#output = model.most_similar(positive = 'hello', topn = 1)   --- torna 'hi'
print(output)

#output = output.data()
#output = output.numpy


#load model defined in class copy_example
try:
    while True:
        domanda = input("You: ")
        domandavettorizzata = model[domanda]

        risposta = model2.most_similar(positive = parola, topn = 1)

        print("ChatBot: " +risposta)
except KeyboardInterrupt:
    print('\nBye bye!')
"""