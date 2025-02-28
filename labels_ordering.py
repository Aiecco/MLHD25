#%%
import pandas as pd

#%%
train_labels = pd.read_csv('Train/train_labels.csv')
val_labels = pd.read_csv('Val/val_labels.csv')
test_labels = pd.read_csv('Test/test_labels.csv')

#%%
train_labels.rename(columns={'male': 'Sex'}, inplace=True)
train_labels['Sex'] = train_labels['Sex'].map({True: 'M', False: 'F'})
train_labels = train_labels[['id', 'Sex', 'boneage']]
train_labels.to_csv('Train/train_labels.csv', index=False)

#%%
val_labels.rename(columns={
    'Image ID': 'id',
    'male': 'Sex',
    'Bone Age (months)': 'boneage'
}, inplace=True)
val_labels['Sex'] = val_labels['Sex'].map({True: 'M', False: 'F'})
val_labels = val_labels[['id', 'Sex', 'boneage']]
val_labels.to_csv('Val/val_labels.csv', index=False)

#%%
test_labels.rename(columns={
    'Case ID': 'id',
    'Ground truth bone age (months)': 'boneage'
}, inplace=True)
test_labels['boneage'] = test_labels['boneage'].astype(int)
test_labels = test_labels[['id', 'Sex', 'boneage']]
test_labels.to_csv('Test/test_labels.csv', index=False)