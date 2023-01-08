import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import datetime
import convLSTM_interpolation

hr_start, hr_end = 0, 1440
n_servers = 3
epsilon = 1e-9
test_fraction = 1/20
valid_fraction = 0
n_epochs = 250
n_lat_grid_lines = 100
n_long_grid_lines = 100
grid_size = 100   # hardcoded, please don't modify
optimizer = "Adam"
features = ['pm1_0']#,'temperature','pressure','humidity']
time_gap = 120
time_steps = (hr_end-hr_start)//time_gap+1
batch_size = 1
real_aqi = 'aqi'
learning_rate = 1e-2
# sliding window hyperparameters
window_size = 5
threshold = 0.5
count_threshold = int(threshold*(window_size*window_size))
test_count = 2
step = 1
window_epochs = 40
window_count = 0
window_method = True

df = pd.read_csv('20-10_all.csv')

columns = ['lat','long','pm1_0','pm2_5','pm10','pressure','temperature','humidity']

# typecasting
for column in columns[2:]:
    df[column] = df[column].astype(float)
df.dateTime = pd.to_datetime(df.dateTime)

# removing anomalies
df = df[(df.lat.astype(int) == 28) &(df.long.astype(int) == 77)]
df = df[(df.pm1_0<=1500) & (df.pm2_5<=1500) & (df.pm10<=1500) & (df.pm1_0>=20) & (df.pm2_5>=30) & (df.pm10>=30)]
df = df[(df.humidity<=60)&(df.humidity>=7)]

print(len(df))

# rounding @5min
df.dateTime = df.dateTime.dt.round(f'{time_gap}min')

# filter time
df = df[(df['dateTime'].dt.day == 20)]
df.dateTime = df.dateTime.dt.hour*60 + df.dateTime.dt.minute
print(f"Before filter : {len(df)}")
df = df[(df.dateTime>=hr_start) & (df.dateTime<=hr_end)]
print(f"After filter : {len(df)}")
df.dateTime -= hr_start

# perform min max normalization
def min_max_normalize(column):
    df[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())

df[real_aqi] = df['pm1_0']*1.0

for column in columns:
    min_max_normalize(column)

for column in columns[:2]:
    df[column] = (df[column]*(grid_size-1)).astype(int)

# for servers, we would create n_server grids of size (4,h,w) and merge them
# 4 : (pm1_0, temperature, pressure, humidity)
# these servers would contain individual data which would be combined through crypten/MPC

df['server'] = np.random.randint(0,n_servers,len(df))

server_grids = [torch.zeros(batch_size,time_steps,len(features),grid_size,grid_size,dtype=torch.float32) for _ in range(n_servers)]   # (b,t,c,h,w)
server_grid_count = [torch.zeros(time_steps,grid_size,grid_size,dtype=torch.int64) for _ in range(n_servers)]

server_aqi_grids = [torch.zeros(batch_size,time_steps,1,grid_size,grid_size,dtype=torch.float32) for _ in range(n_servers)]   # (b,t,c,h,w)

for i, row in df.iterrows():
    for index, j in enumerate(features):
        server_grids[row['server']][0][row['dateTime']//time_gap][0][row['lat']][row['long']] += row[j]
    server_grid_count[row['server']][row['dateTime']//time_gap][row['lat']][row['long']]+=1
    server_aqi_grids[row['server']][0][row['dateTime']//time_gap][0][row['lat']][row['long']] += row[real_aqi]

# to avoid division by 0, won't affect the calculations in any way
decision_server_grid = [server_grid_count[i]==0 for i in range(n_servers)]
for i in range(n_servers):
    server_grid_count[i][decision_server_grid[i]] = 1

# average the values
for i in range(n_servers):
    for j in range(time_steps):
        server_grids[i][0][j][0]/=server_grid_count[i][j]   # they have same dimension hence no broadcasting
        server_aqi_grids[i][0][j][0]/=server_grid_count[i][j]

# combine the values as in real scenario
combined_grid = torch.zeros(batch_size,time_steps,len(features),grid_size,grid_size,dtype=torch.float32)   # train_x
combined_count = torch.zeros(time_steps,grid_size,grid_size,dtype=torch.int64)
combined_aqi = torch.zeros(batch_size,1,grid_size,grid_size,dtype=torch.float32)                # train_y

for i in range(n_servers):
    combined_count += server_grid_count[i]

for i in range(n_servers):
    server_grid_count[i][decision_server_grid[i]]=0

total_over_time_steps = torch.zeros(grid_size,grid_size)

for i in range(n_servers):
    for j in range(time_steps):
        combined_grid[0][j][0] += server_grids[i][0][j][0]*server_grid_count[i][j]/combined_count[j]
        combined_aqi[0][0] += server_aqi_grids[i][0][j][0]*server_grid_count[i][j]
        total_over_time_steps += server_grid_count[i][j]

total_over_time_steps[total_over_time_steps==0]=1
combined_aqi[0][0] /= total_over_time_steps

mask = combined_aqi[0][0]!=0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if window_method:
    model = convLSTM_interpolation.Model((window_size, window_size))
else:
    model = convLSTM_interpolation.Model((grid_size, grid_size))

# move to cuda
print(device)
model = model.to(device)
mask = mask.to(device)

print(f"Total Given Locations : {torch.sum(mask)}")

# Training Begins:
opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
def train(opt,model,train_data,train_labels,mask=mask):
    model.train()
    opt.zero_grad()
    output = model(train_data)
    loss = F.mse_loss(output*mask,train_labels*mask)   # broadcast
    loss.backward()
    opt.step()
    return loss.item(), output

def eval(test_data,test_labels,model,test_indices):
    model.eval()
    output = model(test_data)
    output = output[0][0][test_indices]
    test_labels = test_labels[0][0][test_indices]
    loss = F.mse_loss(output,test_labels)
    print(f"EVALUATION\ntrue\n{test_labels}\npredicted\n{output}")
    return loss.item()

grid_dimension = mask.size()

def sliding_window():
    window_count = 0
    window_test_mse = 0
    window_test_rmse = 0
    total_locations = 0
    arr = []
    d = {}
    for i in range(0, grid_dimension[0]-window_size+1):
        for j in range(0, grid_dimension[1]-window_size+1):
            temp = torch.sum(mask[i:i+window_size,j:j+window_size])
            if temp.item() in d:
                d[temp.item()]+=1
            else:
                d[temp.item()]=1
            if(temp>=count_threshold):
                window_count+=1
                total_locations+=temp
                # create smaller grids
                mask2 = mask[i:i+window_size,j:j+window_size]
                current_indices = torch.nonzero(mask2,as_tuple=True)
                known_locations = len(current_indices[0])
                test_locations1 = random.sample(range(known_locations),test_count)
                test_locations = [None for _ in range(len(current_indices))]
                for k in range(len(current_indices)):
                    test_locations[k] = current_indices[k][test_locations1]
                test_locations = tuple(test_locations)
                mask2[test_locations] = 0
                temp_combined_grid = combined_grid.detach().clone()
                new_combined_grid = temp_combined_grid[:,:,:,i:i+window_size,j:j+window_size]
                new_combined_aqi = combined_aqi[:,:,i:i+window_size,j:j+window_size]
                for k in range(time_steps):
                    for l in range(len(features)):
                        new_combined_grid[0][k][l][test_locations] = 0

                print(new_combined_grid.size())
                print(new_combined_aqi.size())

                # move to cuda
                train_data = new_combined_grid.to(device)
                train_labels = new_combined_aqi.to(device)

                # train
                loss_arr = []
                for k in range(window_epochs):
                    loss_value, output = train(opt,model,train_data,train_labels,mask2)
                    loss_arr.append(loss_value)
                    if k%5==0:
                        print(f"{window_count} ({i},{j},{k}) : {loss_value}")

                arr.append((i,j,train_data,train_labels,test_locations))

    test_indices = set()

    for a in arr:
        i, j, train_data, train_labels, test_locations = a
        eval_loss = eval(train_data, train_labels, model, test_locations)
        print(f"({i},{j},eval) : {eval_loss}")
        window_test_mse += eval_loss*test_count
        # window_test_rmse += np.sqrt(eval_loss)
        temp = [i,j]
        for k in range(test_count):
            temp_arr = []
            for n in range(2):
                temp_arr.append(test_locations[n][k].item()+temp[n])
            test_indices.add(tuple(temp_arr))

    window_test_mse = window_test_mse/len(test_indices)
    window_test_rmse = np.sqrt(window_test_mse)


    print(f"Number of Test Locations : {len(test_indices)}\n{test_indices}")
    print(f"Total windows trained : {window_count}")
    print(f"Average MSE loss : {window_test_mse}")
    print(f"Average RMSE loss : {window_test_rmse}")
    print(f"Total locations trained : {total_locations}")
    print(d)

    # test_values = (torch.tensor([0 for _ in range(len(test_indices))]), torch.tensor([0 for _ in range(len(test_indices))]))
    # for i, j in enumerate(test_indices):
    #     test_values[0][i] = j[0]
    #     test_values[1][i] = j[1]
    #
    # print(test_values)
    # with torch.no_grad():
    #     train_data = combined_grid.to(device)
    #     train_labels = combined_aqi.to(device)
    #     for k in range(time_steps):
    #         for l in range(len(features)):
    #             train_data[0][k][l][test_values] = 0



# train
def normal_train():
    # creating test indices
    indices = torch.nonzero(mask, as_tuple=True)
    known_locations = len(indices[0])
    test_indices1 = random.sample(range(known_locations), int(known_locations * test_fraction))
    test_indices = [None for _ in range(len(indices))]
    for i in range(len(indices)):
        test_indices[i] = indices[i][test_indices1]
    test_indices = tuple(test_indices)
    mask[test_indices] = 0
    # now just need to get indices and make combined_grid to zero at those places.
    for i in range(time_steps):
        for j in range(len(features)):
            combined_grid[0][i][j][test_indices] = 0

    train_data = combined_grid
    train_labels = combined_aqi

    # move to cuda
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    print(combined_grid.size())
    print(combined_aqi.size())

    loss_arr = []
    for i in range(n_epochs):
        loss_value, output = train(opt,model,train_data,train_labels)
        loss_arr.append(loss_value)
        if i % 20==0: print(f"{i}: {loss_arr[i]}")#\n{output[0][0][0]}")

    fig0 = plt.figure(0)
    plt.plot([i for i in range(n_epochs)],loss_arr,label='Loss')
    plt.savefig('loss.png')
    plt.show()

    print(f'RMSE Loss: {np.sqrt(loss_arr[n_epochs-1])}')
    print(f"Evaluation Loss : {np.sqrt(eval(train_data,train_labels,model,test_indices))}")
    print()
    torch.set_printoptions(precision=2,threshold=2e5,linewidth=500,profile='short',sci_mode=False)
    print("Analysing Interpolation")
    print(train_labels[train_labels!=0])
    with torch.no_grad():
        output = model(train_data)
        print(output[train_labels!=0])


sliding_window()
# normal_train()