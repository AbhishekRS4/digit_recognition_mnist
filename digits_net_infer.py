# coding: utf-8
# @author : Abhishek R S

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

from digits_net_utils import read_config_file, get_all_images_labels, get_preprocessed_labels
from digits_net_train import get_placeholders, build_network


param_config_file_name = os.path.join(os.getcwd(), "mnist_config.json")


# return the accuracy score of the model
def get_accuracy_score(labels_actual, labels_predicted):
    return accuracy_score(labels_actual, labels_predicted)


# return the confusion matrix of the predicted labels by the model
def get_confusion_matrix(labels_actual, labels_predicted):
    return confusion_matrix(labels_actual, labels_predicted)


# return the output of the softmax layer of the outputs of the network
def get_softmax_layer(input_tensor, name = 'softmax'):
    prediction = tf.nn.softmax(input_tensor, name = name)
    return prediction


# load the model architecture
def load_model(config, img_pl):
    loaded_model_output = build_network(config, img_pl, training = not(config['TRAINING']))
    return loaded_model_output


def infer():
    print("Reading the Config File..................")
    config = read_config_file(param_config_file_name)
    print("Reading the Config File Completed........")
    print("")

    print("Reading Test Images.....................")
    all_images, all_labels = get_all_images_labels(config, not(config['TRAINING']))
    print("Reading Test Images Completed...........")
    print("")

    print("Preprocessing the data...................")
    all_images = all_images.reshape(all_images.shape[0], all_images.shape[1], all_images.shape[2], config['NUM_CHANNELS'])
    all_labels, all_original_classes = get_preprocessed_labels(all_labels, not(config['TRAINING']))
    all_original_classes = [int(x) for x in all_original_classes]
    print("Preprocessing of the data Completed......")
    print("")

 
    print("Loading the Network.....................")
    
    if config['data_format'] == 'channels_last':
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['NUM_CHANNELS']] + config['TARGET_IMAGE_SIZE']
        all_images = np.transpose(all_images, [0, 3, 1, 2])
 
    img_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = not(config['TRAINING']))
    network_output = load_model(config, img_pl)
    prediction = get_softmax_layer(input_tensor = network_output)
    print("Loading the Network Completed...........")
    print("")

    print("Images shape : " + str(all_images.shape))
    print("Labels shape : " + str(all_labels.shape))
    print("")    

    ss = tf.Session()
    ss.run(tf.global_variables_initializer())
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(config['model_file'][0], config['model_file'][1])) + '-' + str(config['num_epochs']))

    print("")
    print("Inference Started.......................")
    ti = time.time()
    labels_predicted_ohe = ss.run(prediction, feed_dict = {img_pl : all_images})
    ti = time.time() - ti
    print("Inference Completed.....................")
    print("Time Taken for Inference : " +str(ti))
    print("")

    labels_predicted_ohe_tensor = tf.convert_to_tensor(labels_predicted_ohe)
    output_labels = tf.argmax(labels_predicted_ohe_tensor, axis = 1)
    all_labels_predicted = ss.run(output_labels)
    all_labels_predicted = np.array(all_labels_predicted)
    
    print("Accuracy Score of the model : " + str(get_accuracy_score(all_labels, all_labels_predicted)))
    print("")
    print("Confusion Matrix for the prediction : ")
    print(get_confusion_matrix(all_labels, all_labels_predicted))
    print("")
    print("Original Labels : " + str(list(all_original_classes)))
    print("")
    ss.close()


def main():
    infer()

if __name__ == '__main__':
    main()
