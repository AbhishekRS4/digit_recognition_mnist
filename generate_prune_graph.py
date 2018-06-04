import os
import numpy as np
import tensorflow as tf

import network_architecture as na
from digits_net_train import get_placeholders, get_softmax_layer
from digits_net_utils import read_config_file
from digits_net_infer import load_model_vgg, load_model_res, load_model_preact_res

def main():
    param_config_file_name = os.path.join(os.getcwd(), "mnist_config.json")
    config = read_config_file(param_config_file_name)
    model_to_use = config['model_to_use']
    model_directory = config['model_directory'][model_to_use] + str(config['num_epochs'])

    if config['data_format'] == 'channels_last':
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['NUM_CHANNELS']] + config['TARGET_IMAGE_SIZE']
 
    img_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = not(config['TRAINING']))
    training_pl = tf.placeholder(tf.bool)

    if model_to_use == 0:
        network_logits = load_model_vgg(img_pl, training_pl, config)
    elif model_to_use == 1:
        network_logits = load_model_res(img_pl, training_pl, config)
    else:
        network_logits = load_model_preact_res(img_pl, training_pl, config)

    probs_prediction = get_softmax_layer(network_logits)
    network_class_predictions = tf.argmax(probs_prediction, axis = 1, name = 'class_predictions')

    ss = tf.Session()
    ss.run(tf.global_variables_initializer())
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(model_directory, config['model_file'][model_to_use])) + '-' + str(config['num_epochs']))

    frozen_graph = tf.graph_util.convert_variables_to_constants(ss, ss.graph_def, ['class_predictions'])
    tf.train.write_graph(frozen_graph, os.path.join(os.getcwd(), model_directory), 'digits_net_frozen.pb', as_text = False)
    print("Conversion Successful")


if __name__ == '__main__':
    main()
