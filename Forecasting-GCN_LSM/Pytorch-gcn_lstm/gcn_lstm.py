import numpy as np
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from stellargraph.core.utils import calculate_laplacian
torch.manual_seed(0)

class GCN_Layer(nn.Module):
    def __init__(self, grid_size, input_dim, output_dim, adj):
        super().__init__()
        self.d2=grid_size
        self.d3=input_dim
        self.d4=output_dim
        self.A=torch.nn.Parameter(adj)
        #self.A=torch.nn.Parameter(torch.randn(self.d2, self.d2))
        self.kernel=torch.nn.Parameter(torch.randn(self.d3, self.d4))
        self.bias=torch.nn.Parameter(torch.randn(self.d2, 1))
        self.relu=torch.nn.ReLU()
        #torch.nn.init.xavier_uniform_(A, gain=1.0)
        #torch.nn.init.xavier_uniform_(kernel, gain=1.0)

    def forward(self, inputs):
        a=torch.permute(inputs, (0, 2, 1))
        neighbours=torch.matmul(a, self.A)
        h_graph = torch.permute(neighbours, (0, 2, 1))
        output = torch.matmul(h_graph, self.kernel)
        if self.bias is not None:
            output += self.bias
        output = self.relu(output)
        return output

class GCN_LSTM(nn.Module):
    def __init__(self, A, grid_size, seq_len, gc_sizes, lstm_sizes):
        super().__init__()
        self.gc_sizes = gc_sizes
        self.lstm_sizes = lstm_sizes
        self.adj = torch.from_numpy(calculate_laplacian(A) if A is not None else None)
        adj_init = torch.zeros(grid_size, grid_size)

        if self.adj is None:
            nn.init.zeros_(adj_init)
        self.A = self.adj

        for i in range(len(self.gc_sizes)):
            if i==0:
                self.gc_layers = [GCN_Layer(grid_size, seq_len, self.gc_sizes[i], self.A)]
            else:
                self.gc_layers.append(GCN_Layer(grid_size, self.gc_sizes[i-1], self.gc_sizes[i], self.A))
        for i in range(len(self.lstm_sizes)):
            if i==0:
                self.lstm_layers = [nn.LSTM(input_size=grid_size, hidden_size=self.lstm_sizes[i], batch_first=True)]
            else:
                self.lstm_layers.append(nn.LSTM(input_size=self.lstm_sizes[i-1], hidden_size=self.lstm_sizes[i], batch_first=True))
        ##MANUAL
        self.dropout=torch.nn.Dropout()
        self.dense = torch.nn.Linear(self.lstm_sizes[-1], grid_size)
        self.activation = torch.nn.Sigmoid()

    def forward(self, inputs):
        h_layer = inputs
        for layer in self.gc_layers:
            h_layer = layer(h_layer) 
        out_gcn=torch.permute(h_layer, [0, 2, 1])
        out_lstm = out_gcn
        for layer in self.lstm_layers:
            out_lstm = layer(out_lstm)[0] #, (self.initial_hidden, self.initial_cell)
        out = torch.permute(out_lstm, (1, 0, 2))
        out = out[-1]
        #print(out.shape, out)
        out = self.dropout(out)
        out = self.dense(out)
        #print(output)
        out = self.activation(out)
        return out
