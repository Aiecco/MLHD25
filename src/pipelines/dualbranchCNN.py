import os

from src.Models.DualBranchCNN import DualBranchCNN
from src.Utils.filelabels_search import filelabels_search
from src.Utils.save_tensors import save_tensors
from src.preprocessing.preprocess import preprocess_dataset
from src.preprocessing.preprocess_images import plot_tensor_image


def process_folder(folder_path, tensors_dict, filepath):
    print(f'Preprocessing dataset:\n{os.path.basename(folder_path)} set')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            id_img = int(filename.split('.')[0])
            image_path = os.path.join(folder_path, filename)
            tensors_dict[id_img] = {} #creo un dizionario per ogni id_img
            tensors_dict[id_img]['tensor'] = preprocess_dataset(image_path)
            #tensors_dict[id_img]['sex'], tensors_dict[id_img]['boneage'] = filelabels_search(
            #    os.path.join(folder_path, f'../{filepath}'), id_img)

    return tensors_dict

def pipeline_dualbranchCNN(preprocess=False):
    # Percorsi delle cartelle
    test_path = 'data/Test/test_samples'
    train_path = 'data/Train/train_samples'
    val_path = 'data/Val/validation_samples'

    if preprocess:
        test_tensors = process_folder(test_path, {}, 'test_labels.csv')
        train_tensors = process_folder(train_path, {}, 'train_labels.csv')
        val_tensors = process_folder(val_path, {}, 'val_labels.csv')

        # Plot e save per test
        if test_tensors:
            for id_img, data in test_tensors.items():
                if data['tensor'] is not None:
                    #plot_tensor_image(data['tensor'])
                    save_tensors(os.path.join(os.path.dirname(test_path), 'tensors'), id_img, data)


        # Plot e save per train
        if train_tensors:
            for id_img, data in train_tensors.items():
                if data['tensor'] is not None:
                    #plot_tensor_image(data['tensor'])
                    save_tensors(os.path.join(os.path.dirname(train_path), 'tensors'), id_img, data)

        # Plot e save per validation
        if val_tensors:
            for id_img, data in val_tensors.items():
                if data['tensor'] is not None:
                    #plot_tensor_image(data['tensor'])
                    save_tensors(os.path.join(os.path.dirname(val_path), 'tensors'), id_img, data)