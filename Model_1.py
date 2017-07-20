import string
import re
import sys
import os
import time
from vector import vector
import gensim
from gensim.models import Word2Vec
#from CornellData import CornellData
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

class Model_1(nn.Module):

    def __init__(self, input_size, sos_idx, eos_idx, encoder_layers = 1, lstm_size = 128):
        # Call parent
        super(Model_1, self).__init__()
        # Set attributes
        self.is_cuda = False
        self.input_size = 302
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
        super(Model_1, self).cuda()

    def forward(self, x, h, target_as_input):
        #model = gensim.models.KeyedVectors.load_word2vec_format('/media/daniele/AF56-12AA/GoogleNews-vectors-negative300.bin', binary=True)
        model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
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
            input = torch.zeros(1, 1, self.input_size)
            input = Variable(input)
            if self.is_cuda:
                input = input.cuda()
            h = h_0
            c = c_0
            # Initialize list of outputs at each time step
            output = []
            # Process until EOS is found or limit is reached
            for i in range (0, 100):
                o, hc=self.decoder(input, (h,c))
                h,c=hc
                o2=self.dec_to_output(o.view(-1,self.lstm_size))
                o2=o2.view(batch_size, 1, -1)
                output.append(o2)
                output_x=o2.data.squeeze()
                output_vec=output_x.numpy()
                output_vec=output_vec[0:300]
                vettor=model.most_similar(positive=[output_vec], topn=1)[0][0]
                vettor=model[vettor]
                vettor=torch.from_numpy(vettor)
                padding = torch.zeros(2)
                vettor=torch.cat((vettor, padding), 0)
                vettor=vettor.unsqueeze(0)
                vettor=vettor.unsqueeze(0)
                input=vettore
                input = Variable(input)
                if self.is_cuda:
                   input = input.cuda(async = True)
            x = torch.cat(output, 1)
        return x, h_0