# coding: utf-8
# @author : Abhishek R S

import math
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from digits_net_utils import init, get_all_images_labels, read_config_file, get_preprocessed_labels, get_train_validation_set, get_accuracy_score


param_config_file_name = os.path.join(os.getcwd(), "mnist_config.json")


# return ELU activation function
def get_elu_activation(input_tensor, name = 'elu'):
    output_tensor = tf.nn.elu(input_tensor, name = name)
    return output_tensor


# return MaxPool2D layer
def get_maxpool2d_layer(input_tensor, pool_size, strides, padding, data_format, name = 'maxpool'):
    output_tensor = tf.layers.max_pooling2d(inputs = input_tensor, pool_size = pool_size, strides = strides, padding = padding, data_format = data_format, name = name)
    return output_tensor


# return Convolution2D layer
def get_conv2d_layer(input_tensor, num_filters, kernel_size, strides, padding, data_format, name = 'conv'):
    output_tensor = tf.layers.conv2d(inputs = input_tensor, filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format, name = name)
    return output_tensor


# return the dense layer
def get_dense_layer(input_tensor, num_neurons, name = 'dense'):
    return tf.layers.dense(input_tensor, units = num_neurons, name = name)


# return the flattened features
def get_flattened_features(input_tensor, name = 'flatten'):
    return tf.layers.flatten(input_tensor, name = name)


# return the dropout layer
def get_dropout_layer(input_tensor, rate = 0.5, training = False, name = 'dropout'):
    return tf.layers.dropout(inputs = input_tensor, rate = rate, training = training, name = name)


# return the output of the softmax layer
def get_softmax_layer(input_tensor, name = 'softmax'):
    prediction = tf.nn.softmax(input_tensor, name = name)
    return prediction


# return the loss function which has to be minimized
def get_loss_function(groundtruth_labels, predicted_logits, name = 'categorical_cross_entropy'):
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = groundtruth_labels, logits = predicted_logits, name = name))
    return loss_function


# return the optimizer which has to be used to minimize the loss function
def get_optimizer(learning_rate, loss_function):
    adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_function)
    return adam_optimizer


# return the placeholder based on the flag
def get_placeholders(img_placeholder_shape, training = False, lbl_placeholder_shape = None):
    img_pl = tf.placeholder(tf.float32, shape = img_placeholder_shape)
    # set the image placeholder

    if training:
        # set the label placeholder in the training phase
        label_pl = tf.placeholder(tf.float32, shape = lbl_placeholder_shape)
        return (img_pl, label_pl)

    return img_pl


# save the trained model
def save_model(session, model_directory, model_file, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), os.path.join(model_directory, model_file)), global_step = (epoch + 1))


# build the network and return the output layer
def build_network(config, img_pl, training = False):
    
    # encoder stage 1
    output_tensor = get_conv2d_layer(input_tensor = img_pl, num_filters = config['num_kernels'][0], kernel_size = config['kernel_size'], strides = config['strides'], padding = config['padding'], data_format = config['data_format'], name = 'conv1_1')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu1_1')
    output_tensor = get_conv2d_layer(input_tensor = output_tensor, num_filters = config['num_kernels'][0], kernel_size = config['kernel_size'], strides = config['strides'], padding = config['padding'], data_format = config['data_format'], name = 'conv1_2')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu1_2')
    output_tensor = get_maxpool2d_layer(input_tensor = output_tensor, pool_size = config['pool_size'], strides = config['pool_size'], padding = config['padding'], data_format = config['data_format'], name = 'maxpool_1')
    

    # encoder stage 2
    output_tensor = get_conv2d_layer(input_tensor = output_tensor, num_filters = config['num_kernels'][1], kernel_size = config['kernel_size'], strides = config['strides'], padding = config['padding'], data_format = config['data_format'], name = 'conv2_1')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu2_1')
    output_tensor = get_conv2d_layer(input_tensor = output_tensor, num_filters = config['num_kernels'][1], kernel_size = config['kernel_size'], strides = config['strides'], padding = config['padding'], data_format = config['data_format'], name = 'conv2_2')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu2_2')
    output_tensor = get_maxpool2d_layer(input_tensor = output_tensor, pool_size = config['pool_size'], strides = config['pool_size'], padding = config['padding'], data_format = config['data_format'], name = 'maxpool_2')
    
    # flattened features
    output_tensor = get_flattened_features(input_tensor = output_tensor, name = 'flatten')
    output_tensor = get_dense_layer(input_tensor = output_tensor, num_neurons = config['dense_layer_neurons'][0], name = 'dense_1')
    output_tensor = get_dropout_layer(input_tensor = output_tensor, rate = config["dropout_rate"], training = training, name = 'dropout')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu3')

    output_tensor = get_dense_layer(input_tensor = output_tensor, num_neurons = config['NUM_CLASSES'], name = 'logits')
    return output_tensor

def batch_train():

    print("Reading the config file..................")
    config = read_config_file(param_config_file_name)
    print("Reading the config file completed........")
    print("")

    print("Initializing.............................")
    model_directory = config['model_directory'] + str(config['num_epochs'])
    init(model_directory)
    print("Initializing completed...................")
    print("")

    print("Reading train data.......................")
    all_images, all_labels = get_all_images_labels(config, bool(config['TRAINING']))
    print("Reading train data completed.............")
    print("")

    print("Preprocessing the data...................")
    all_images = all_images.reshape(all_images.shape[0], all_images.shape[1], all_images.shape[2], config['NUM_CHANNELS'])
    all_labels = get_preprocessed_labels(all_labels, bool(config['TRAINING']))
    train_images, train_labels, valid_images, valid_labels = get_train_validation_set(all_images, all_labels) 
    print("Preprocessing of the data completed......")
    print("")

 
    print("Building the network.....................")
     
    if config['data_format'] == 'channels_last': 
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['NUM_CHANNELS']] + config['TARGET_IMAGE_SIZE']
        train_images = np.transpose(train_images, [0, 3, 1, 2])
        valid_images = np.transpose(valid_images, [0, 3, 1, 2])
    
    LABEL_PLACEHOLDER_SHAPE = [None] + [config['NUM_CLASSES']]
    img_pl, lbl_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = bool(config['TRAINING']), lbl_placeholder_shape = LABEL_PLACEHOLDER_SHAPE)
    
    network_output = build_network(config, img_pl, training = bool(config['TRAINING']))
    loss = get_loss_function(lbl_pl, network_output)
    optimizer = get_optimizer(config['learning_rate'], loss)
 
    print("Building the network completed...........")
    print("")
    
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_batches = int(math.ceil(train_images.shape[0] / float(batch_size)))
    
    print("Train Images shape : " + str(train_images.shape))
    print("Train Labels shape : " + str(train_labels.shape))
    print("Validation Images shape : " + str(valid_images.shape))
    print("Validation Labels shape : " + str(valid_labels.shape))
    print("Number of epochs to train : " + str(num_epochs))
    print("Batch size : " + str(batch_size))
    print("Number of batches : " + str(num_batches))
    print("")

    print("Training the Network.....................")
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    train_loss_per_epoch = list()
    valid_loss_per_epoch = list()

    for epoch in range(num_epochs):
        ti = time.time()
        temp_loss_per_epoch = 0
        train_images, train_labels = shuffle(train_images, train_labels) 
        for batch_id in range(num_batches):
            batch_images = train_images[batch_id * batch_size : (batch_id + 1) * batch_size]
            batch_labels = train_labels[batch_id * batch_size : (batch_id + 1) * batch_size]
            
            _, loss_per_batch = ss.run([optimizer, loss], feed_dict = {img_pl : batch_images, lbl_pl : batch_labels})
            temp_loss_per_epoch += (batch_labels.shape[0] * loss_per_batch)
        ti = time.time() - ti
        loss_validation_set = ss.run(loss, feed_dict = {img_pl : valid_images, lbl_pl : valid_labels})
        train_loss_per_epoch.append(temp_loss_per_epoch)
        valid_loss_per_epoch.append(loss_validation_set)
        print("Epoch : " + str(epoch+1) + "/" + str(num_epochs) + ", time taken : " + str(ti) + " sec.")
        print("Avg. training loss : " + str(temp_loss_per_epoch / train_images.shape[0]))
        print("Avg. validation loss : " + str(loss_validation_set))
        print("")
    
    print("Training the Network Completed...........")
    print("")
    
    print("Saving the model.........................")
    save_model(ss, model_directory, config['model_file'], epoch)
    train_loss_per_epoch = np.array(train_loss_per_epoch)
    valid_loss_per_epoch = np.array(valid_loss_per_epoch)
    
    train_loss_per_epoch = np.true_divide(train_loss_per_epoch, train_images.shape[0])
   
    losses_dict = dict()
    losses_dict['train_loss'] = train_loss_per_epoch
    losses_dict['valid_loss'] = valid_loss_per_epoch

    np.save(os.path.join(os.getcwd(), os.path.join(model_directory, config['model_metrics'])), (losses_dict))
    print("Saving the model Completed...............")
    print("")
    ss.close()

def main():
    batch_train()

if __name__ == '__main__':
    main()

