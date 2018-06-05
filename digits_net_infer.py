# coding: utf-8
# @author : Abhishek R S

import os
import time
import numpy as np
import tensorflow as tf

import network_architecture as na
from digits_net_utils import read_config_file, get_all_images_labels, get_preprocessed_labels, get_accuracy_score, get_confusion_matrix
from digits_net_train import get_placeholders, get_softmax_layer


param_config_file_name = os.path.join(os.getcwd(), "mnist_config.json")


# load the network based on vgg encoder
def load_model_vgg(img_pl, training_pl, config):
    net_arch = na.Network_Architecture(img_pl, config['kernel_size'], config['num_kernels'], config['strides'], config['data_format'], config['padding'], config['pool_size'], training_pl, config['dense_layer_neurons'], config['NUM_CLASSES'], config['dropout_rate'])
    net_arch.vgg_encoder()
    logits = net_arch.logits

    return logits

# load the network based on normal residual encoder
def load_model_res(img_pl, training_pl, config):
    net_arch = na.Network_Architecture(img_pl, config['kernel_size'], config['num_kernels'], config['strides'], config['data_format'], config['padding'], config['pool_size'], training_pl, config['dense_layer_neurons'], config['NUM_CLASSES'], config['dropout_rate'], config['pool_size'])
    net_arch.residual_encoder()
    logits = net_arch.logits

    return logits

# load the network based on pre-activation residual encoder
def load_model_preact_res(img_pl, training_pl, config):
    net_arch = na.Network_Architecture(img_pl, config['kernel_size'], config['num_kernels'], config['strides'], config['data_format'], config['padding'], config['pool_size'], training_pl, config['dense_layer_neurons'], config['NUM_CLASSES'], config['dropout_rate'], config['pool_size'])
    net_arch.preactivation_residual_encoder()
    logits = net_arch.logits

    return logits


# run inference on test set
def infer():
    print("Reading the Config File..................")
    config = read_config_file(param_config_file_name)
    model_to_use = config['model_to_use']
    model_directory = config['model_directory'][model_to_use] + str(config['num_epochs'])
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
    training_pl = tf.placeholder(tf.bool)

    if model_to_use == 0:
        network_logits = load_model_vgg(img_pl, training_pl, config)
    elif model_to_use == 1:
        network_logits = load_model_res(img_pl, training_pl, config)
    else:
        network_logits = load_model_preact_res(img_pl, training_pl, config)

    probs_prediction = get_softmax_layer(input_tensor = network_logits)
    print("Loading the Network Completed...........")
    print("")

    print("Images shape : " + str(all_images.shape))
    print("Labels shape : " + str(all_labels.shape))
    print("")    

    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    # load the model parameters
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(model_directory, config['model_file'][model_to_use])) + '-' + str(config['num_epochs']))

    print("")
    print("Inference Started.......................")
    ti = time.time()
    probs_predicted = ss.run(probs_prediction, feed_dict = {img_pl : all_images, training_pl : not(config['TRAINING'])})
    ti = time.time() - ti
    print("Inference Completed.....................")
    print("Time Taken for Inference : " +str(ti))
    print("")

    probs_predicted_tensor = tf.convert_to_tensor(probs_predicted)
    output_labels = tf.argmax(probs_predicted_tensor, axis = 1)
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
