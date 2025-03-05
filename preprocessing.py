#%% md
# Label Preprocessing
#%%
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
#%% md
# Image Preprocessing
#%%
import mediapipe as mp
import numpy as np
import random
import cv2
import os
#%%
def clahe3(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result

def flipAndRotate(img):
    if (random.random() > 0.5): img = cv2.flip(img, 1)
    if (random.random() > 0.5): img = cv2.flip(img, -1)
    rnd = random.random()
    if rnd < 0.25: img = cv2.rotate(img, cv2.ROTATE_180)
    elif rnd < 0.5: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rnd < 0.75: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
#%% raw

''' # experiment which we chose not to use
def handRec(img):
    global suc
    result = Hands.process(img)
    h, w, c = img.shape
    hand_landmarks = result.multi_hand_landmarks
    x_max = 0
    x_min = w
    y_min = h
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
            # cv2.rectangle(img, (x_min, y_min), (x_max, h), (0, 255, 0), 2)
            # mp.solutions.drawing_utils.draw_landmarks(img, handLMs, mphands.HAND_CONNECTIONS)
        offset = int((h * offset_percent) / 100)
        y_min_new = y_min - offset
        x_min_new = x_min - offset
        x_max_new = x_max + offset
        # pad = int(
        #     (abs(y_min_new - h) - abs(x_min_new - x_max_new)) / 2)
        # if (pad > 0):
        #     x_min_new -= pad
        #     x_max_new += pad
        # if (pad < 0):
        #     y_min_new -= pad
        if (y_min_new < 0):
            y_min_new = 0
        if (x_min_new < 0):
            x_min_new = 0
        if (x_max_new > w):
            x_max_new = w
        # print(f"%s) offest: %s  | crop(y1,y2,x1,x2): %s, %s, %s, %s "%(count, offset, y_min_new, y_max_new, x_min_new, x_max_new))
        suc += 1
        return img[y_min_new:h, x_min_new:x_max_new]
    else:
        # print(f"%s) coudln't find the hand!"%count)
        return img

def getx(img):
    w, h, c = img.shape
    result = Hands.process(img)
    mx, wx = 0, 0
    if (result.multi_hand_landmarks):
        for hand_lm in result.multi_hand_landmarks:
            mx = hand_lm.landmark[hands.HandLandmark.MIDDLE_FINGER_TIP].x * w
            wx = hand_lm.landmark[hands.HandLandmark.WRIST].x * w
    return wx, mx
#%%
hands = mp.solutions.hands
Hands = hands.Hands()
offset_percent = 5  # offset percentage for croping the detected hand
rotation_percent = 20  # offset percentage for hand straightening
count = 0  # count of images which has been processed
suc = 0  # count of images wihcn successfully found a hand in it
'''

#%% md
### Preprocess Validation and Test
#%%
image_dir_val = 'Val/val_samples'
image_dir_test = 'Test/test_samples'

output_dir_val = 'Val/pp_val_samples' # new
output_dir_test = 'Test/pp_test_samples' # new


os.makedirs(output_dir_val, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)
#%%
for filename in os.listdir(image_dir_val): #val
    img = cv2.imread(os.path.join(image_dir_val, filename))
    if img is None:
        print("ERROR:", filename)
    else:
        # img = handRec(img)  # detect the hand for cropping
        # img = clahe3(img)  # adjust brightness and contrast

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0  # Normalize
        img = (img * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir_val, filename), img)  # Save the processed image
        print(f"img {filename} processed")
#%%
for filename in os.listdir(image_dir_test): #test
    img = cv2.imread(os.path.join(image_dir_test, filename))
    if img is None:
        print("ERROR:", filename)
    else:
        # img = handRec(img)  # detect the hand for cropping
        # img = clahe3(img)  # adjust brightness and contrast

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0  # Normalize
        img = (img * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir_test, filename), img)  # Save the processed image
        print(f"img {filename} processed")

#%% md
### Preprocess Train (we add flip and rotation as augmentation)
#%%
image_dir_train = 'Train/train_samples'
output_dir_train = 'Train/pp_train_samples' # new

os.makedirs(output_dir_train, exist_ok=True)
#%%
for filename in os.listdir(image_dir_train):
    img_path = os.path.join(image_dir_train, filename)
    img = cv2.imread(img_path)

    if img is None:
        print("ERROR loading:", filename)
        continue

    img = flipAndRotate(img)  # rotate before recognition/cropping

    # img = handRec(img)
    # img = clahe3(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale

    # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0  # Normalize
    img = (img * 255).astype(np.uint8)

    output_path = os.path.join(output_dir_train, filename)
    cv2.imwrite(output_path, img)
    print(f"img {filename} processed")


# --------------------------- LOADING
# paths

base_path = ""

train_images_path = os.path.join(base_path, "Train/pp_train_samples")
val_images_path = os.path.join(base_path, "Val/pp_val_samples")
test_images_path = os.path.join(base_path, "Test/pp_test_samples")

train_labels = pd.read_csv(os.path.join(base_path, "Train/train_labels.csv"))
val_labels = pd.read_csv(os.path.join(base_path, "Val/val_labels.csv"))
test_labels = pd.read_csv(os.path.join(base_path, "Test/test_labels.csv"))


# loading
target_size = (224, 224)

def load(image_folder, labels):
    X, y = [], []
    for _, row in labels.iterrows():
        image_id = row["id"]
        img_path = os.path.join(image_folder, f"{image_id}.png")

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale images
        print(f"loading {img_path}")

        img = cv2.resize(img, target_size)
        img = np.expand_dims(img, axis=-1)  # add channel dimension
        X.append(img)
        y.append(row["boneage"])  # response

    return np.array(X), np.array(y)


X_train, y_train = load(train_images_path, train_labels)
X_val, y_val = load(val_images_path, val_labels)
X_test, y_test = load(test_images_path, test_labels)


# saving
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
