# coding: utf-8
# @author : Abhishek R S

import os
import json
import numpy as np
from scipy.misc import imread, imresize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# read the json file and return the content
def read_config_file(json_file_name):
	# open and read the json file
    config = json.load(open(json_file_name))

    # return the content
    return config


# returns the label of the image
def get_label(image_path):
	# return the label of the current image given the path of the image
	return image_path.split(os.sep)[-2]


# returns all the image file paths
def get_all_image_paths(path_to_directory):
    file_paths = list()

    for root, directories, files in os.walk(path_to_directory):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    
    return file_paths


# returns either train or test data based on the flag
def get_all_images_labels(config, training = False):
	# all_images is a list holding all the images
    all_images = list()

    # all_labels is a list holding all the correspoding labels
    all_labels = list()
    
    # based on the flag, set whether to read train or test data
    if training:
    	images_directory_path = os.path.join(os.getcwd(), os.path.join(config['train_images_path'][0], config['train_images_path'][1]))
    else:
    	images_directory_path = os.path.join(os.getcwd(), os.path.join(config['test_images_path'][0], config['test_images_path'][1]))

    all_image_paths = get_all_image_paths(images_directory_path)

    # read the image and get the corresponding label and append it to the appropriate list
    for img_file in all_image_paths:
        try:
            temp_img = imresize(imread(img_file), config['TARGET_IMAGE_SIZE'])
            temp_label = get_label(img_file)

            all_images.append(temp_img)
            all_labels.append(temp_label)
        except IOError:
            print("File not an image or image file is corrupt")
            
    # convert the data into numpy array and return it
    return np.array(all_images), np.array(all_labels)


# returns labels in one-hot encoding manner
def get_preprocessed_labels(all_labels, training = False):
    lbl_encoder = LabelEncoder()
    all_labels = lbl_encoder.fit_transform(all_labels)
    
    all_labels = all_labels.reshape(all_labels.shape[0], 1)
    
    if training == False:
        return all_labels, lbl_encoder.classes_

    lbl_onehot_encoder = OneHotEncoder(categorical_features = [0])
    all_labels = lbl_onehot_encoder.fit_transform(all_labels).toarray()

    return all_labels