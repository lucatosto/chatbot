# Define options
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
opt.save_every = 20
# Model options
opt.encoder_layers = 1
opt.lstm_size = 1024
# Test options
import sys
opt.model = sys.argv[1] if len(sys.argv) > 1 else None
opt.test = sys.argv[2] if len(sys.argv) > 2 else None
# Backend options
opt.no_cuda = True #False if use cuda gpu
# Imports
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

from vector import vector

# Create datasets
a = vector()
dataset_totale= a.vettorizzazione()
train_dataset=dataset_totale[:20]
test_dataset=dataset_totale[21:25]

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
#mseloss
mseloss = torch.nn.MSELoss(size_average=False)# AL POSTO DI LSTM_SOFTMAX_LOSS

# Setup CUDA
#if not opt.no_cuda:
#    model.cuda()

# Monitoring options
update_every = 100
cnt = 0
# Start training
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

model_options = {'input_size': train_dataset[0][0].size(1), 'sos_idx': sos_idx, 'eos_idx': eos_idx, 'encoder_layers': opt.encoder_layers, 'lstm_size': opt.lstm_size}
model = Model1(**model_options)
optimizer = torch.optim.SGD(model.parameters(), lr = opt.learning_rate, momentum = opt.momentum, weight_decay = opt.weight_decay)
model.train()


try:
    for epoch in range(1, opt.epochs+1):
        # Adjust learning rate for SGD
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Initialize loss/accuracy
        sum_loss = 0 # training only
        n_train = 0 # for averaging
        sum_length_accuracy = 0 # test only
        sum_match_accuracy = 0 # test only
        n_test = 0 # for averaging

        for split in ["train"]:
            # Set mode
            # Process all training batches

            for conv in train_dataset:
                h = torch.zeros(opt.encoder_layers, 1, opt.lstm_size)

                for b in range(0, len(conv)-1):
                    input = conv[b]
                    target = conv[b+1]
                    input = input.unsqueeze(0)
                    target = target.unsqueeze(0)
                    if not opt.no_cuda:
                        input = input.cuda(async = True)
                        target = target.cuda(async = True)
                    # Wrap for autograd
                    input = Variable(input, volatile = (split != "train"))#mettendo l'input qui dentro autograd fa si che venga wrappato e reso adatto per l'allenamento
                    target_as_input = Variable(target[:, :-1, :], volatile = (split != "train"))
                    target_as_target = Variable(target[:, 1:, :], volatile = (split != "train"))
                    # Forward (use target as decoder input for training)
                    output, h_nuova = model(input, Variable(h), target_as_input)
                    h = h_nuova.data.clone()
                    # Compute loss (training only)
                    loss = mseloss(output, target_as_target) if split == "train" else Variable(torch.Tensor([-1]))
                    sum_loss += loss.data[0]
                    n_train += 1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Show results every once in a while
                    cnt += 1
                    if cnt % update_every == 0:
                        print("Epoch {0}: L = {1:.4f}, LA = {2:.4f}, MA = {3:.4f}\r".format(epoch, sum_loss/n_train, sum_length_accuracy/n_test if n_test > 0 else -1, sum_match_accuracy/n_test if n_test > 0 else -1), end = '')
        # Print info at the end of the epoch
        print("Epoch {0}: L = {1:.4f}, LA = {2:.4f}, MA = {3:.4f}".format(epoch, sum_loss/n_train, sum_length_accuracy/n_test if n_test > 0 else -1, sum_match_accuracy/n_test if n_test > 0 else -1))
        # Save checkpoint
        if epoch % opt.save_every == 0:
            # Build file name
            checkpoint_path = "checkpoint-" + repr(epoch) + ".pth"
            # Write data
            torch.save({'model_options': model_options, 'model_state': model.state_dict()}, checkpoint_path)
except KeyboardInterrupt:
    print("Interrupted")
