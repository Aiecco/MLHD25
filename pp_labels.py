import pandas as pd
#%%
train_labels = pd.DataFrame(pd.read_csv('Train/train_labels.csv'))
val_labels = pd.DataFrame(pd.read_csv('Val/val_labels.csv'))
test_labels = pd.DataFrame(pd.read_csv('Test/test_labels.csv'))
#%%
# uniform column names with dictionaries
train_labels.rename(columns={"id": "id", "boneage": "boneage", "male": "male"}, inplace=True) # we use this as the standard
val_labels.rename(columns={"Image ID": "id", "Bone Age (months)": "boneage"}, inplace=True)
test_labels.rename(columns={"Case ID": "id", "Sex": "male", "Ground truth bone age (months)": "boneage"}, inplace=True)

# convert male column to boolean
test_labels["male"] = test_labels["male"].replace({"M": True, "F": False})

# test boneage is float but the others are int
train_labels["boneage"] = train_labels["boneage"].astype(float)
val_labels["boneage"] = val_labels["boneage"].astype(float)

col_order = ["id", "boneage", "male"]
train_labels = train_labels[col_order]
val_labels = val_labels[col_order]
test_labels = test_labels[col_order]

# verify uniformity
val_labels.info()
test_labels.info()
#%%
train_labels.to_csv("Train/train_labels.csv", index=False)
val_labels.to_csv("Val/val_labels.csv", index=False)
test_labels.to_csv("Test/test_labels.csv", index=False)