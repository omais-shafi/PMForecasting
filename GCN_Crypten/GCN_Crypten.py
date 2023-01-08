#!/usr/bin/env python
# coding: utf-8

# In[1]:
import logging


# In[ ]:


logging.getLogger().setLevel(logging.INFO)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import datetime
import math
import timeit
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# In[2]:


import crypten
crypten.init()
torch.set_num_threads(1)


# In[3]:


hr_start, hr_end = 660, 720
n_servers = 7
epsilon = 1e-3
test_fraction = 1/20
valid_fraction = 1/10
n_epochs = 100
n_lat_grid_lines = 100
n_long_grid_lines = 100


# In[4]:


from torch import Tensor
from torch.nn import Linear


# In[5]:


import datetime
#Put the file location
df = pd.read_csv('../Pollution-Prediction-GNN-main/20-10_all.csv')

#type casting
df.pm1_0 = df.pm1_0.astype(float)
df.pm2_5 = df.pm2_5.astype(float)
df.pm10 = df.pm10.astype(float)
df.lat = round(round(5*df.lat.astype(float),2)/5.0,3)
df.long= round(round(5*df.long.astype(float),2)/5.0,3)
df.pressure = df.pressure.astype(float)
df.temperature = df.temperature.astype(float)
df.humidity = df.humidity.astype(float)
df.dateTime = pd.to_datetime(df.dateTime)

print(len(df))
# Ensuring Delhi region and removing outliers from data
df = df[(df.lat.astype(int) == 28) &(df.long.astype(int) == 77)]
df = df[(df.pm1_0<=1500) & (df.pm2_5<=1500) & (df.pm10<=1500) & (df.pm1_0>=20) & (df.pm2_5>=30) & (df.pm10>=30)]
df = df[(df.humidity<=60)&(df.humidity>=7)]
print(len(df))


# In[6]:


# rounding @15min
df.dateTime = df.dateTime.dt.round('15min')

# only PM10
dfHour = df[(df['dateTime'].dt.day == 20)][['dateTime','lat','long','pm1_0']]

# TODO: Add buffer and consider train-data for range
lat_range = {'min': dfHour.lat.min(), 'max': dfHour.lat.max()}
long_range = {'min': dfHour.long.min(), 'max': dfHour.long.max()}

dfHour['lat_grid'] = dfHour.apply(lambda row: int(n_lat_grid_lines*(row.lat-lat_range['min'])/(lat_range['max']-lat_range['min'])), axis=1 )
dfHour['long_grid'] = dfHour.apply(lambda row: int(n_long_grid_lines*(row.long-long_range['min'])/(long_range['max']-long_range['min'])), axis=1 )
dfHour['lat_grid'] = dfHour['lat_grid'].astype(float).astype(int)
dfHour['long_grid'] = dfHour['long_grid'].astype(float).astype(int)
del dfHour['lat']
del dfHour['long']

# TODO: use time as a feature as well
# converting time to minutes and selecting a one hour slot
dfHour.dateTime = dfHour.dateTime.dt.hour*60 + dfHour.dateTime.dt.minute
dfHour = dfHour[(dfHour.dateTime>=hr_start) & (dfHour.dateTime<=hr_end)] 

del dfHour['dateTime']


# In[7]:


dfHour.long_grid.value_counts()


# In[8]:


dfHour.lat_grid.value_counts()


# In[9]:


dfHour['server'] = np.random.randint(0, n_servers, len(dfHour))
dfHour = dfHour.reset_index(drop=True)

grid = {}
count = 0
count2 = 0
for i,row in dfHour.iterrows():
#     print(f"{int(row.lat_grid)} {int(row.long_grid)}")
    if(not f"{int(row.lat_grid)} {int(row.long_grid)}" in grid):
        grid[f"{int(row.lat_grid)} {int(row.long_grid)}"] = count
        count += 1
        count2 += 1
#     if(not f"{int(row.lat_grid + 1)} {int(row.long_grid)}" in grid):
#         grid[f"{int(row.lat_grid + 1)} {int(row.long_grid)}"] = count
#         count += 1
#     if(not f"{int(row.lat_grid)} {int(row.long_grid + 1)}" in grid):
#         grid[f"{int(row.lat_grid)} {int(row.long_grid + 1)}"] = count
#         count += 1
#     if(not f"{int(row.lat_grid - 1)} {int(row.long_grid)}" in grid):
#         grid[f"{int(row.lat_grid - 1)} {int(row.long_grid)}"] = count
#         count += 1
#     if(not f"{int(row.lat_grid)} {int(row.long_grid - 1)}" in grid):
#         grid[f"{int(row.lat_grid)} {int(row.long_grid - 1)}"] = count
#         count += 1

print(count)
print(count2)

adj = np.zeros((count, count))
aqi = np.zeros((n_servers, count))
mask = np.zeros((n_servers, count))

for i,row in dfHour.iterrows():
    mask_ser_grid = mask[int(row.server)][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]]
    aqi[int(row.server)][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] *= mask_ser_grid/(mask_ser_grid+1)
    aqi[int(row.server)][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] += row.pm1_0/(mask_ser_grid+1)
    mask[int(row.server)][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] += 1
    
    if(f"{int(row.lat_grid + 1)} {int(row.long_grid)}" in grid):
        adj[grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid + 1)} {int(row.long_grid)}"]] = 1
    if(f"{int(row.lat_grid)} {int(row.long_grid + 1)}" in grid):
        adj[grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid + 1)}"]] = 1
    if(f"{int(row.lat_grid - 1)} {int(row.long_grid)}" in grid):
        adj[grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid - 1)} {int(row.long_grid)}"]] = 1
    if(f"{int(row.lat_grid)} {int(row.long_grid - 1)}" in grid):
        adj[grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid - 1)}"]] = 1
    
    if(f"{int(row.lat_grid + 1)} {int(row.long_grid)}" in grid):
        adj[grid[f"{int(row.lat_grid + 1)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] = 1
    if(f"{int(row.lat_grid)} {int(row.long_grid + 1)}" in grid):
        adj[grid[f"{int(row.lat_grid)} {int(row.long_grid + 1)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] = 1
    if(f"{int(row.lat_grid - 1)} {int(row.long_grid)}" in grid):
        adj[grid[f"{int(row.lat_grid - 1)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] = 1
    if(f"{int(row.lat_grid)} {int(row.long_grid - 1)}" in grid):
        adj[grid[f"{int(row.lat_grid)} {int(row.long_grid - 1)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] = 1
    
#     adj[grid[f"{int(row.lat_grid + 1)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid + 1)} {int(row.long_grid)}"]] = 1
#     adj[grid[f"{int(row.lat_grid)} {int(row.long_grid + 1)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid + 1)}"]] = 1
#     adj[grid[f"{int(row.lat_grid - 1)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid - 1)} {int(row.long_grid)}"]] = 1
#     adj[grid[f"{int(row.lat_grid)} {int(row.long_grid - 1)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid - 1)}"]] = 1
    adj[grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]][grid[f"{int(row.lat_grid)} {int(row.long_grid)}"]] = 1


# In[10]:


mask = torch.tensor(mask)
adj = torch.tensor(adj)
aqi = torch.tensor(aqi)


# In[11]:


test_indices = random.sample(range(aqi.shape[1]), int(aqi.shape[1] * test_fraction))
aqi_train = aqi * 1
aqi_train[:, test_indices] = 0

mask_train = mask * 1
mask_train[:, test_indices] = 0

adj_train = adj.clone().detach()
adj_train[:, test_indices] = 0


# In[12]:


def train_valid_split(train_data, rank):
    mult = torch.ones(train_data['aqi'].shape)
    mult2 = torch.ones(train_data['mask'].shape)
    mult3 = torch.ones(train_data['adj'].shape)
    
    if rank == 0:
        valid_indices = random.sample(range(train_data['aqi'].shape[0]), int(train_data['aqi'].shape[0] * valid_fraction))

        mult[valid_indices, :] = 0

        mult2[valid_indices] = 0
        
        mult3[:, valid_indices] = 0
    
    mult = crypten.cryptensor(mult, src = 0) 
    
    aqi_train_valid = train_data['aqi'] * mult
    
    mult2 = crypten.cryptensor(mult2, src=0) 
    
    mask_train_valid = train_data['mask'] * mult2
    mask_train_valid = mask_train_valid > 0.1
#     crypten.print(train_data['mask'].sum().get_plain_text())
#     crypten.print(mask_train_valid.sum().get_plain_text())
    
    mult3 = crypten.cryptensor(mult3, src=0).get_plain_text() 
    new_adj = train_data['adj'].clone().detach()
    new_adj = new_adj * mult3
        
    train_valid_data = {
        'adj': new_adj,
        'aqi': aqi_train_valid,
        'mask': mask_train_valid
    }
    return train_valid_data


# In[13]:


class GCNConv(crypten.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        temp_weight = torch.FloatTensor(in_features, out_features)
        if bias:
            temp_bias = torch.FloatTensor(out_features)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(temp_weight, temp_bias)

    def reset_parameters(self, temp_weight, temp_bias):
        stdv = 1. / math.sqrt(temp_weight.size(1))
        
        temp_weight.data.uniform_(-stdv, stdv)
        self.register_parameter('weight', temp_weight)
        
        if temp_bias is not None:
            temp_bias.data.uniform_(-stdv, stdv)
            self.register_parameter('bias', temp_bias)
            

    def forward(self, input, adj):
        #  Support becomes N X output_features
        support = input.matmul(self.weight)
        temp = crypten.cryptensor(adj)
        
        D = (1 + temp.sum(dim=1).unsqueeze(-1)).reciprocal()
        #  Output is N X output_features
        output = D * temp.matmul(support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('                + str(self.in_features) + ' -> '                + str(self.out_features) + ')'


# In[14]:


# TODO/ASK: too many parameters
class Net(crypten.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1Mean = GCNConv(2,8)        
        self.conv2Mean = GCNConv(8,8)
        
        self.lin3 = crypten.nn.Linear(8, 1)
    
    def lrelu(self, x, alpha):
        temp1 = ((x < 0) * alpha) + (x > 0)
        return temp1 * x
    
    def forward(self, data):
        x, adj = crypten.cat([data['aqi'],data['mask'].unsqueeze(-1)],1), data['adj']
        x = self.lrelu(self.conv1Mean(x, adj), 0.1)
#         crypten.print(x.get_plain_text())
        x = self.lrelu(self.conv2Mean(x, adj), 0.1)
#         crypten.print(x.get_plain_text())
        x = self.lin3(x)
#         crypten.print(x.get_plain_text())
        return x.squeeze()


# In[15]:


def train(net, data_train, combined_y_enc, rank):
    net.train() # Change to training mode
    net.encrypt()
    loss = crypten.nn.MSELoss() # Choose loss functions

    
    crypten_optimizer = crypten.optim.SGD(net.parameters(), lr = 0.0001)
    
    # Set parameters: learning rate, num_epochs
#     learning_rate = 0.001
    num_epochs = 5
      
    
    # Train the model: SGD on encrypted data
    for i in range(num_epochs):
        
        # set gradients to zero
        net.zero_grad()
        
        data_valid = train_valid_split(data_train, rank=rank)  
        
        # forward pass
        output = net(data_valid)
#         crypten.print(data_train['mask'])
        loss_value = loss(output * data_train['mask'], combined_y_enc * data_train['mask'])
        
        # perform backward pass
        loss_value.backward()

        # update parameters
        crypten_optimizer.step()
            
        # examine the loss after each epoch
        # crypten.print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))


# In[16]:


import crypten.mpc as mpc
import crypten.communicator as comm
from crypten.communicator.communicator import _logging
from crypten.common.approximations import ConfigManager
num_parties = n_servers
torch.manual_seed(42)
model = Net()


# In[17]:
tic = 0

@mpc.run_multiprocess(world_size=num_parties)
def run_encrypted_training():
    global model, tic
    comm.get().set_verbosity(True)

    # Get rank of current process
    rank = comm.get().get_rank()
    
    with ConfigManager("reciprocal_nr_iters", 12):

        # Parties' data
        x = aqi_train[rank].unsqueeze(-1)
        train_mask = mask_train[rank].unsqueeze(-1)
        y = aqi_train[rank]

        # Combined data
        combined_mask_enc = crypten.cryptensor(train_mask, src = 0)
        combined_x_enc = crypten.cryptensor(x, src = 0) * combined_mask_enc
        combined_y_enc = crypten.cryptensor(y, src = 0) * combined_mask_enc.t()

        adj_matrix = adj_train

        # Get features and adjacency matrix
        for i in range(1, num_parties):       
            x_enc = crypten.cryptensor(x, src = i)
            y_enc = crypten.cryptensor(y, src = i)
            mask_enc = crypten.cryptensor(train_mask, src = i)

            combined_x_enc = combined_x_enc + x_enc * mask_enc
            combined_y_enc = combined_y_enc + y_enc * mask_enc.t()
            combined_mask_enc = crypten.cat([combined_mask_enc, mask_enc], dim=1)

        # Add epsilon to prevent 0/0
        normalise = (combined_mask_enc.sum(dim=1, keepdim=True) + epsilon) / 10

        # Combined x encrypted -> average of all
        combined_x_enc = combined_x_enc / 10
        combined_x_enc = combined_x_enc / normalise

        combined_y_enc = combined_y_enc / 10 
        combined_y_enc = (combined_y_enc / normalise.t()).squeeze()

        # Take or to get final mask
        combined_mask_enc = (combined_mask_enc.sum(dim=1, keepdim=True) > 0).squeeze()

        train_data = {
            'adj': adj_matrix,
            'aqi': combined_x_enc,
            'mask': combined_mask_enc
        }

    #     crypten.print(torch.max(normalise.get_plain_text()))
        # crypten.print(combined_mask_enc.sum().get_plain_text())
#         crypten.print(combined_y_enc.get_plain_text())


        if rank == 0:
            comm.get().reset_communication_stats()
            tic = timeit.default_timer()

        if rank == 0:
            toc = timeit.default_timer()
            comm.get().print_communication_stats()
            print("Total time:{}".format(toc - tic))

        net = model
        net.encrypt()

        train(net, train_data, combined_y_enc, rank)
        output = net(train_data).get_plain_text()
        model = net.decrypt()

        if rank == 0:
            toc = timeit.default_timer()
            comm.get().print_communication_stats()
            print("Total time:{}".format(toc - tic))
        
    return output, model

# tic = timeit.default_timer()
output_t = run_encrypted_training()
output, model = output_t[0]
# print(output)
# print((aqi*mask).sum(axis=0)/(mask.sum(axis=0) + epsilon))
# output, model = output[0]


# In[18]:


# for i in model.named_parameters():
#     print(i)


# In[19]:


aqi_mean = (aqi*mask).sum(axis=0)/(mask.sum(axis=0) + epsilon)
mask_mean = (mask).sum(axis=0)
mask_mean[mask_mean > 0] = 1
mask_mean_train = mask_mean * 1
mask_mean_train[test_indices] = 0
aqi_mean_train = aqi_mean * 1
aqi_mean_train[test_indices] = 0
train_data = {
    'adj': adj,
    'aqi': aqi_mean_train.view(-1,1),
    'mask': mask_mean_train
}
def eval(output = output, aqi_mean=aqi_mean, test_indices=test_indices):
    test_output = (torch.reshape(output,(-1,)) * mask_mean)[test_indices]
    true_test_output = aqi_mean[test_indices]
    test_loss = F.mse_loss(test_output, true_test_output)
    test_loss *= len(train_data['mask'])/torch.sum(train_data['mask'])
    print(test_output)
    print(true_test_output)
    return test_loss.item()

# # In[13]:


# In[20]:


print(eval())


# In[21]:


# model.lrelu(crypten.cryptensor([-3.57, -0.1542, 2.3141, 4.213, 0]), 0.1).get_plain_text()


# In[22]:


# def train(opt=opt, model=model, train_data=train_data):
#     model.train()
#     opt.zero_grad()
#     train_valid_data = train_valid_split(train_data)
#     output = model(train_valid_data)
# #     print(output[:5])
# #     print(train_data['aqi'][0:5])
#     loss = F.mse_loss(torch.reshape(output,(-1,)) * train_data['mask'],torch.reshape(train_data['aqi'],(-1,)))
#     loss *= len(train_data['mask'])/torch.sum(train_data['mask'])
#     loss.backward()
#     opt.step()
#     return loss.item()

# def eval(model=model, train_data=train_data, aqi_mean=aqi_mean, test_indices=test_indices):
#     model.eval()
#     output = model(train_data)
#     test_output = (torch.reshape(output,(-1,)) * torch.FloatTensor(mask_mean))[test_indices]
#     true_test_output = torch.FloatTensor(aqi_mean)[test_indices]
#     test_loss = F.mse_loss(test_output, true_test_output)
# #     test_loss = F.mse_loss(30*torch.ones(len(test_indices)) * torch.FloatTensor(mask_mean)[test_indices], true_test_output)
#     test_loss *= len(train_data['mask'])/torch.sum(train_data['mask'])
#     print(test_output)
#     print(true_test_output)
#     return test_loss.item()


# In[23]:


# actual_test_indices = np.intersect1d(np.where(mask_mean > 0), test_indices)


# # In[24]:


# for ind in range(len(actual_test_indices)):
#     print(actual_test_indices[ind])
#     print(np.where(adj[actual_test_indices[ind]]))
#     print(aqi_mean[actual_test_indices[ind]])
#     print(aqi_mean[np.where(adj[actual_test_indices[ind]])])


# # In[25]:


# eval()


# # In[26]:


# # TODO: too little data and too little input features


# # In[27]:


# torch.cat((torch.ones((5,1)), torch.ones((5))), axis=1).shape


# # In[ ]:


# mult = torch.ones(train_data['aqi'].shape)


# # In[ ]:


# mult


# In[ ]:





# In[ ]:




