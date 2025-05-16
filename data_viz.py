# inspecting the json file
import os
import json 
import nibabel as nib # to load the nifti files
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # to plot the images
from matplotlib.colors import ListedColormap # to create custom colormap
# import garage, data_prep # custom library to visualize the images



# loading the dataset, convert to numpy array and exploring the dataset

def load_case(case_id):
    # load the image (4D array: 240, 240, 155, 4)
    image = np.array(nib.load(f'BrainTumour_dataset/imagesTr/{case_id}.nii.gz').get_fdata())
    # load the label (3D array: 240, 240, 155)
    label = np.array(nib.load(f'BrainTumour_dataset/labelsTr/{case_id}.nii.gz').get_fdata())

    return image, label

"""
The colors correspond to each class.
Red is edema
Green is a non-enhancing tumor
Blue is an enhancing tumor.
"""




if __name__ == "__main__":
    json_path = 'BrainTumour_dataset/dataset.json' # path to the json file
    # loading the json file
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)

    print(dataset_info.keys()) # checking the keys of the json file
    print(dataset_info['modality']) # checking the labels of the json file

    print(dataset_info['tensorImageSize'])# checking the size of the images
    print(dataset_info['numTraining']) # checking the number of training images
    print(dataset_info['numTest']) # checking the number of test images
    print("TensorFlow:", tf.__version__)
   





