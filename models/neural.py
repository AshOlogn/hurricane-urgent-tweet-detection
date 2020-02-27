import numpy as np
import pandas as pd
import random
import time
from itertools import product

# MIT License

# Copyright (c) 2017 Ben Trevett

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import warnings;
warnings.filterwarnings('ignore');

class CNN_Tweet(nn.Module):
    def __init__(self, embeddings, n_filters, filter_sizes, n_classes, dropout):
        
        super(CNN_Tweet, self).__init__()
        
        #length of the word and character embeddings
        word_embedding_dim = embeddings.shape[1]
        
        #architecture
        self.word_embedding = nn.Embedding.from_pretrained(embeddings).cuda()
        
        self.word_convs = [nn.Conv2d(in_channels = 1, 
                               out_channels = n_filters, 
                               kernel_size = (f_size, word_embedding_dim)).cuda() 
                     for f_size in filter_sizes]
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_classes).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.softmax = nn.Softmax().cuda()
        
    def forward(self, word_indices):
        
        embedded = self.word_embedding(word_indices)
        embedded = embedded.unsqueeze(1)
        
        conved = [F.tanh(conv(embedded)).squeeze(3) for conv in self.word_convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.softmax(self.fc(cat))
    
    def predict(self, tweet):
        return np.argmax(self.forward(tweet).detach().cpu().numpy())
