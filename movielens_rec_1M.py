#%%
# import libraries
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.data import Dataset
import pandas as pd

USE_TINY_DATASET = True
remove_percentage = 0.9 # higer percentage, less sparse. not equavalent to 100k!

#%%
# build tiny dataset
def remove_least_common(file_path, user_percentage=0.2, item_percentage=0.2):
    # Read the data
    data = pd.read_csv(file_path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')

    # Remove duplicate data and keep the highest rating
    data = data.sort_values('rating', ascending=False).drop_duplicates(['user_id', 'item_id']).sort_index()

    # Calculate the occurrence count of each user and item
    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()

    # Calculate the number of users and items to remove
    num_users_to_remove = int(len(user_counts) * user_percentage)
    num_items_to_remove = int(len(item_counts) * item_percentage)

    # Find the least common users and items
    users_to_remove = user_counts.nsmallest(num_users_to_remove).index
    items_to_remove = item_counts.nsmallest(num_items_to_remove).index

    # Remove these users and items
    data = data[~data['user_id'].isin(users_to_remove)]
    data = data[~data['item_id'].isin(items_to_remove)]

    return data

if USE_TINY_DATASET:
    file_path = 'dataset/ml-1m/ratings.dat'
    data = remove_least_common(file_path, user_percentage=remove_percentage, item_percentage=remove_percentage)

    # Check the sparsity of data
    # num_users = data['user_id'].unique().shape[0]
    # num_items = data['item_id'].unique().shape[0]
    # sparsity = float(data.shape[0]) / float(num_users * num_items)
    # print('Sparsity: {:.2f}%'.format(sparsity * 100))

    # Save the new dataset to a CSV file
    data.to_csv('dataset/ml-1m/.rating_tiny.dat', header=False, index=False)

    # Read the file and replace comma separators with ::
    with open('dataset/ml-1m/.rating_tiny.dat', 'r') as r, open('dataset/ml-1m/rating_tiny.dat', 'w') as w:
        for line in r:
            w.write(line.replace(',', '::'))

#%%
# build interaction matrix
def build_interaction_matrix(file_path, test_percentage=0.2):
    # Read the data
    data = pd.read_csv(file_path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')

    # Remove duplicate data and keep the highest rating
    data = data.sort_values('rating', ascending=False).drop_duplicates(['user_id', 'item_id']).sort_index()

    # Create a dataset object
    dataset = Dataset()

    # Fit the data and create mappings for users and items
    dataset.fit(users=data['user_id'].unique(), items=data['item_id'].unique())

    # Sort by timestamp
    data = data.sort_values('timestamp')

    # Calculate the test set size for each user
    data['rank'] = data.groupby('user_id').cumcount()
    user_test_size = data.groupby('user_id').size() * test_percentage
    user_test_size = user_test_size.apply(np.ceil).astype(int)

    # Assign the last few ratings of each user as the test set
    data['split'] = 'train'
    data.loc[data['rank'] < data['user_id'].map(user_test_size), 'split'] = 'test'

    # Build interaction matrices for the training and test sets
    train_interactions, _ = dataset.build_interactions(data.loc[data['split'] == 'train', ['user_id', 'item_id']].values)
    test_interactions, _ = dataset.build_interactions(data.loc[data['split'] == 'test', ['user_id', 'item_id']].values)

    return train_interactions, test_interactions

if USE_TINY_DATASET:
    file_path = 'dataset/ml-1m/rating_tiny.dat'
else:
    file_path = 'dataset/ml-1m/ratings.dat'
train_interactions, test_interactions = build_interaction_matrix(file_path, test_percentage=0.1)

# Check the sparsity of data
num_users, num_items = train_interactions.shape
sparsity = 1 - float(train_interactions.nnz) / float(num_users * num_items)
print('Sparsity: {:.2f}%'.format(sparsity * 100))

#%%
# create model and train
model = LightFM(no_components=20, learning_rate=0.02, loss='warp')
model.fit(train_interactions, epochs=5, verbose=True, num_threads=4)

#%%
# evaluate model
train_precision = precision_at_k(model, train_interactions, k=10).mean()
test_precision = precision_at_k(model, test_interactions, k=10, train_interactions=train_interactions).mean()
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))

train_auc = auc_score(model, train_interactions).mean()
test_auc = auc_score(model, test_interactions, train_interactions=train_interactions).mean()
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

#%%