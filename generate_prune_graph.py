import os
import numpy as np
import tensorflow as tf
from digits_net_train import build_network, get_placeholders
from digits_net_utils import read_config_file
from digits_net_infer import get_softmax_layer

def main():
    param_config_file_name = os.path.join(os.getcwd(), "mnist_config.json")
    config = read_config_file(param_config_file_name)

    if config['data_format'] == 'channels_last':
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['NUM_CHANNELS']] + config['TARGET_IMAGE_SIZE']
 
    img_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = not(config['TRAINING']))

    network_logits = build_network(config, img_pl, training = not(config['TRAINING']))
    network_probabilities = get_softmax_layer(network_logits)
    network_class_predictions = tf.argmax(network_probabilities, axis = 1, name = 'class_predictions')

    ss = tf.Session()
    ss.run(tf.global_variables_initializer())
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(config['model_file'][0], config['model_file'][1])) + '-' + str(config['num_epochs']))

    frozen_graph = tf.graph_util.convert_variables_to_constants(ss, ss.graph_def, ['class_predictions'])
    tf.train.write_graph(frozen_graph, os.path.join(os.getcwd(), config['model_file'][0]), 'digits_net_frozen.pb', as_text = False)
    print("Conversion Successful")


if __name__ == '__main__':
    main()
