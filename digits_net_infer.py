# coding: utf-8
# @author : Abhishek R S

import os
import time
import numpy as np
import tensorflow as tf

from digits_net_utils import read_config_file, get_all_images_labels, get_preprocessed_labels, get_accuracy_score, get_confusion_matrix
from digits_net_train import get_placeholders, build_network, get_softmax_layer


param_config_file_name = os.path.join(os.getcwd(), "mnist_config.json")


# load the model architecture
def load_model(config, img_pl):
    loaded_model_output = build_network(config, img_pl, training = not(config['TRAINING']))
    return loaded_model_output


def infer():
    print("Reading the Config File..................")
    config = read_config_file(param_config_file_name)
    model_directory = config['model_directory'] + str(config['num_epochs'])
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
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(model_directory, config['model_file'])) + '-' + str(config['num_epochs']))

    print("")
    print("Inference Started.......................")
    ti = time.time()
    logits_predicted = ss.run(prediction, feed_dict = {img_pl : all_images})
    ti = time.time() - ti
    print("Inference Completed.....................")
    print("Time Taken for Inference : " +str(ti))
    print("")

    logits_predicted_tensor = tf.convert_to_tensor(logits_predicted)
    output_labels = tf.argmax(logits_predicted_tensor, axis = 1)
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
