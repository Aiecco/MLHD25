import os
import tensorflow as tf

from src.preprocessing.preprocess import preprocess_dataset


def process_folder(folder_path, tensors_dict):
    print(f'Preprocessing dataset:\n{os.path.basename(folder_path)} set')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            id_img = int(filename.split('.')[0])
            image_path = os.path.join(folder_path, filename)
            tensors_dict[id_img] = {}  # creo un dizionario per ogni id_img
            tensors_dict[id_img]['tensor'], tensors_dict[id_img]['heated'] = preprocess_dataset(image_path)
            tensors_dict[id_img]['tensor'] = tf.convert_to_tensor(tensors_dict[id_img]['tensor'], dtype=tf.float32)
            tensors_dict[id_img]['heated'] = tf.convert_to_tensor(tensors_dict[id_img]['heated'], dtype=tf.float32)

    return tensors_dict