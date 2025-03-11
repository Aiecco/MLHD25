import pandas as pd


def train_labels(path):
    data = pd.read_csv(path)
    data.rename(columns={'male': 'Sex'}, inplace=True)
    if data['Sex'].any not in ['M', 'F']:
        data['Sex'] = data['Sex'].map({True: 'M', False: 'F'})
    data = data[['id', 'Sex', 'boneage']]
    data.to_csv('data/Train/train_labels.csv', index=False)


def val_labels(path):
    data = pd.read_csv(path)
    data.rename(columns={
        'Image ID': 'id',
        'male': 'Sex',
        'Bone Age (months)': 'boneage'
    }, inplace=True)
    if data['Sex'].any not in ['M', 'F']:
        data['Sex'] = data['Sex'].map({True: 'M', False: 'F'})
    data = data[['id', 'Sex', 'boneage']]
    data.to_csv('data/Val/val_labels.csv', index=False)


def test_labels(path):
    data = pd.read_csv(path, delimiter=';')
    data.rename(columns={
        'Case ID': 'id',
        'Ground truth bone age (months)': 'boneage'
    }, inplace=True)
    data['boneage'] = data['boneage'].astype(int)
    data = data[['id', 'Sex', 'boneage']]
    data.to_csv('data/Test/test_labels.csv', index=False)