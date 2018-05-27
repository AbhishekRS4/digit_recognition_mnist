# coding: utf-8
# @author : Abhishek R S

import os
import json
import numpy as np
from scipy.misc import imread, imresize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# read the json file and return the content
def read_config_file(json_file_name):
    # open and read the json file
    config = json.load(open(json_file_name))

    # return the content
    return config


# create the model directory if not present
def init(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# return the label of the image
def get_label(image_path):
    # return the label of the current image given the path of the image
    return image_path.split(os.sep)[-2]


# return all the image file paths
def get_all_image_paths(path_to_directory):
    file_paths = list()

    for root, directories, files in os.walk(path_to_directory):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    
    return file_paths


# return either train or test data based on the flag
def get_all_images_labels(config, training = False):
    # all_images is a list holding all the images
    all_images = list()

    # all_labels is a list holding all the correspoding labels
    all_labels = list()
    
    # based on the flag, set whether to read train or test data
    if training:
    	images_directory_path = os.path.join(os.getcwd(), os.path.join(config['TRAIN_IMAGES_PATH'][0], config['TRAIN_IMAGES_PATH'][1]))
    else:
    	images_directory_path = os.path.join(os.getcwd(), os.path.join(config['TEST_IMAGES_PATH'][0], config['TEST_IMAGES_PATH'][1]))

    all_image_paths = get_all_image_paths(images_directory_path)

    # read the image and get the corresponding label and append it to the appropriate list
    for img_file in all_image_paths:
        try:
            temp_img = imresize(imread(img_file), config['TARGET_IMAGE_SIZE'])
            temp_label = get_label(img_file)

            all_images.append(temp_img)
            all_labels.append(temp_label)
        except IOError:
            print("File is either corrupted or not an image")
            
    # convert the data into numpy array and return it
    return np.array(all_images), np.array(all_labels)


# return labels in one-hot encoding manner
def get_preprocessed_labels(all_labels, training = False):
    lbl_encoder = LabelEncoder()
    all_labels = lbl_encoder.fit_transform(all_labels)
    
    all_labels = all_labels.reshape(all_labels.shape[0], 1)
    
    if training == False:
        return all_labels, lbl_encoder.classes_

    lbl_onehot_encoder = OneHotEncoder(categorical_features = [0])
    all_labels = lbl_onehot_encoder.fit_transform(all_labels).toarray()

    return all_labels


# split into train and validation set
def get_train_validation_set(all_images, all_labels, validation_size = 0.04):
    train_images, valid_images, train_labels, valid_labels = train_test_split(all_images, all_labels, test_size = validation_size, random_state = 4)
    
    return (train_images, train_labels, valid_images, valid_labels)


# return the accuracy score of the model
def get_accuracy_score(labels_groundtruth, labels_predicted):
    return accuracy_score(labels_groundtruth, labels_predicted)


# return the confusion matrix of the predicted labels by the model
def get_confusion_matrix(labels_groundtruth, labels_predicted):
    return confusion_matrix(labels_groundtruth, labels_predicted)
