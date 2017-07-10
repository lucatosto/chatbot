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


model2 = torch.load('checkpoint-2.pth')
print(model2)
