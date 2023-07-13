#%%
# import packages
import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd

#%%
# MF from scratch using PyTorch
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
    def forward(self, user, item):
        pred = self.user_factors(user) * self.item_factors(item)
        pred = pred.sum(1) + self.user_bias(user).squeeze() + self.item_bias(item).squeeze()
        return pred

#%%
# load data
file_path = 'dataset/ml-1m/rating_tiny.dat'
data = pd.read_csv(file_path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')

# remove duplicate data and keep the highest rating
data = data.sort_values('rating', ascending=False).drop_duplicates(['user_id', 'item_id']).sort_index()

# to consecutive user_id and item_id
user_ids = data['user_id'].unique()
item_ids = data['item_id'].unique()
user2idx = {o:i for i,o in enumerate(user_ids)}
item2idx = {o:i for i,o in enumerate(item_ids)}
data['user_id'] = data['user_id'].apply(lambda x: user2idx[x])
data['item_id'] = data['item_id'].apply(lambda x: item2idx[x])
n_users = len(user_ids)
n_items = len(item_ids)

# split train and test by timestamp by user
data = data.sort_values('timestamp')
data['rank'] = data.groupby('user_id').cumcount()
user_test_size = data.groupby('user_id').size() * 0.1
user_test_size = user_test_size.apply(np.ceil).astype(int)
data['split'] = 'train'
data.loc[data['rank'] < data['user_id'].map(user_test_size), 'split'] = 'test'

train = data.loc[data['split'] == 'train', ['user_id', 'item_id', 'rating']]
test = data.loc[data['split'] == 'test', ['user_id', 'item_id', 'rating']]

#%%
# initialize model, loss and optimizer
model = MatrixFactorization(n_users, n_items, n_factors=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#%%
# train
model.train()
n_epochs = 10

for epoch in range(n_epochs):
    users = Variable(users)
    items = Variable(items)
    ratings = Variable(ratings)
    
    outputs = model(users, items)
    loss = criterion(outputs, ratings)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, n_epochs, loss.item()))

#%%
# evaluate precision@k
model.eval()
with torch.no_grad():
    outputs = model(users, items)
    _, predicted = torch.topk(outputs, 3, dim=0)
    print(predicted.t())

#%%