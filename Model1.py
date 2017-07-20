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

class Model1(nn.Module):

    def __init__(self, input_size, sos_idx, eos_idx, encoder_layers = 1, lstm_size = 128):
        # Call parent
        super(Model1, self).__init__()
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
        super(Model1, self).cuda()

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
            for i in range(0, 50): #this must be dependent on dataset
                # Get decoder output at this time step
                o, hc = self.decoder(input, (h, c))
                h, c = hc
                o = o.view(-1, self.lstm_size)
                #print(x.data.size())
                o = self.dec_to_output(o)



                #o2=o.view(batch_size, 1, -1)
                o2=o.data.squeeze()
                o2=o2.data.numpy()






                #o2=o.data[0].numpy()
                o2=o2[0:300]
                #print(o2.shape)
                o3=model.similar_by_vector(o2, topn=1)[0][0]
                #o3 = model.similar_by_vector(positive=[o2], topn=1)[0][0]
                o3 = model[o3] #prende la parola codificata dal modello
                # Compute log-softmax
                #o2 = F.log_softmax(o2)
                # View as sequence and add to outputs
                print(o2)
                #o2=model[o2]
                #o2=np.ndarray([1,300])
                o3=torch.from_numpy(o3)
                padding = torch.zeros(2)
                o3 = torch.cat((o3, padding), 0)
                o3=o3.unsqueeze(0)
                o3=o3.unsqueeze(0)
                #o2 = o2.view(batch_size, 1, -1)# da fare

                output.append(o3)
                input=o3 # da eliminare?
                # Compute predicted outputs
                #output_idx = o2.data[0].max(2)[1].squeeze()
                # Check all words are in EOS
                # Compute input for next step    l'uscita del primo hidden layer
                input = torch.zeros(batch_size, 1, self.input_size)
                """
                for j in range(0, batch_size):
                    input[j, 0, output_idx[j]] = 1
                """
                input = Variable(input)
                if self.is_cuda:
                    input = input.cuda(async = True)
            # Concatenate all log-softmax outputs
            x = torch.cat(output, 1)
        return x, h_0
