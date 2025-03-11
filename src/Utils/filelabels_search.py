import pandas as pd


def filelabels_search(folder, id):
    df = pd.read_csv(folder, sep=';')
    try:
        df = df[df['id'] == id]
    except KeyError:
        df = pd.read_csv(folder)
        df = df[df['id'] == id]

    return df['sex'].iloc[0], df['boneage'].iloc[0]
