# coding: utf-8
# @author : Abhishek R S

import math
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from digits_net_utils import get_all_images_labels, read_config_file, get_preprocessed_labels



param_config_file_name = os.path.join(os.getcwd(), "mnist_config.json")


# return ELU activation function
def get_elu_activation(input_tensor, name = 'elu'):
    output_tensor = tf.nn.elu(input_tensor, name = name)
    return output_tensor


# return ReLU activation function
def get_relu_activation(input_tensor, name = 'relu'):
    output_tensor = tf.nn.relu(input_tensor, name = name)
    return output_tensor


# return Batch Normalization layer
def get_batchnorm_layer(input_tensor, trainable = False, name = 'batchnorm'):
    output_tensor = tf.layers.batch_normalization(input_tensor, trainable = trainable, name = name)
    return output_tensor


# return Max_Pool2D layer
def get_maxpool2d_layer(input_tensor, pool_size, strides, padding, data_format, name = 'maxpool'):
    output_tensor = tf.layers.max_pooling2d(inputs = input_tensor, pool_size = pool_size, strides = strides, padding = padding, data_format = data_format, name = name)
    return output_tensor


# return Concolution2D layer
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


# return the loss function which has to be minimized
def get_loss_function(actual_labels, calculated_logits, name = 'categorical_cross_entropy'):
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = actual_labels, logits = calculated_logits, name = name))
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
def save_model(session, checkpoint_dir, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), os.path.join(checkpoint_dir[0], checkpoint_dir[1])), global_step = (epoch+1))


# build the network and return the output layer
def build_network(config, img_pl, training = False):
    
    
    output_tensor = get_conv2d_layer(input_tensor = img_pl, num_filters = config['NUM_KERNELS'][0], kernel_size = config['KERNEL_SIZE'], strides = config['STRIDES'], padding = config['PADDING'], data_format = config['DATA_FORMAT'], name = 'conv1_1')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu1_1')
    output_tensor = get_conv2d_layer(input_tensor = output_tensor, num_filters = config['NUM_KERNELS'][0], kernel_size = config['KERNEL_SIZE'], strides = config['STRIDES'], padding = config['PADDING'], data_format = config['DATA_FORMAT'], name = 'conv1_2')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu1_2')
    output_tensor = get_maxpool2d_layer(input_tensor = output_tensor, pool_size = config['POOL_SIZE'], strides = config['POOL_SIZE'], padding = config['PADDING'], data_format = config['DATA_FORMAT'], name = 'maxpool_1')
    
    output_tensor = get_conv2d_layer(input_tensor = output_tensor, num_filters = config['NUM_KERNELS'][1], kernel_size = config['KERNEL_SIZE'], strides = config['STRIDES'], padding = config['PADDING'], data_format = config['DATA_FORMAT'], name = 'conv2_1')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu2_1')
    output_tensor = get_conv2d_layer(input_tensor = output_tensor, num_filters = config['NUM_KERNELS'][1], kernel_size = config['KERNEL_SIZE'], strides = config['STRIDES'], padding = config['PADDING'], data_format = config['DATA_FORMAT'], name = 'conv2_2')
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu2_2')
    output_tensor = get_maxpool2d_layer(input_tensor = output_tensor, pool_size = config['POOL_SIZE'], strides = config['POOL_SIZE'], padding = config['PADDING'], data_format = config['DATA_FORMAT'], name = 'maxpool_2')
    
    output_tensor = get_flattened_features(input_tensor = output_tensor, name = 'flatten')
    output_tensor = get_dense_layer(input_tensor = output_tensor, num_neurons = config['DENSE_LAYER_NEURONS'][0], name = 'dense_1')
    output_tensor = get_dropout_layer(input_tensor = output_tensor, rate = config["DROPOUT_RATE"], training = training, name = "dropout")
    output_tensor = get_elu_activation(input_tensor = output_tensor, name = 'elu3')

    output_tensor = get_dense_layer(input_tensor = output_tensor, num_neurons = config['NUM_CLASSES'], name = 'logits')
    return output_tensor

def batch_train():
    print("Reading the Config File......................")
    config = read_config_file(param_config_file_name)
    print("Reading the Config File Completed............")
    print("")

    print("Reading Train Images.....................")
    all_images, all_labels = get_all_images_labels(config, bool(config['TRAINING']))
    print("Reading Train Images Completed...........")
    print("")

    print("Preprocessing the data...................")
    all_images = all_images.reshape(all_images.shape[0], all_images.shape[1], all_images.shape[2], config['NUM_CHANNELS'])
    all_labels = get_preprocessed_labels(all_labels, bool(config['TRAINING']))
    print("Preprocessing of the data Completed......")
    print("")

    print("Images shape : " + str(all_images.shape))
    print("Labels shape : " + str(all_labels.shape))
 
    print("Building the Network.....................")
    
    IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
    LABEL_PLACEHOLDER_SHAPE = [None] + [config['NUM_CLASSES']]
    
    img_pl, lbl_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = bool(config['TRAINING']), lbl_placeholder_shape = LABEL_PLACEHOLDER_SHAPE)
    
    network_output = build_network(config, img_pl, training = bool(config['TRAINING']))
    loss = get_loss_function(lbl_pl, network_output)
    optimizer = get_optimizer(config['learning_rate'], loss)
    
    print("Building the Network Completed...........")
    print("")
    
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_batches = int(math.ceil(all_images.shape[0] / float(batch_size)))
    
    print("Number of Epochs to Train : " + str(num_epochs))
    print("Batch size : " + str(batch_size))
    print("Number of Batches : " + str(num_batches))
    print("")

    print("Training the Network.....................")
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    loss_per_epoch = list()

    for epoch in range(num_epochs):
        ti = time.time()
        temp_loss_per_epoch = 0
        all_images, all_labels = shuffle(all_images, all_labels) 
        for batch_id in range(num_batches):
            batch_images = all_images[batch_id * batch_size : (batch_id + 1) * batch_size]
            batch_labels = all_labels[batch_id * batch_size : (batch_id + 1) * batch_size]
            
            _, loss_per_batch = ss.run([optimizer, loss], feed_dict = {img_pl : batch_images, lbl_pl : batch_labels})
            print("Avg. Loss for batch : " + str(batch_id+1) + " = " + str(loss_per_batch))
            temp_loss_per_epoch += (batch_size * loss_per_batch)
        ti = time.time() - ti
        loss_per_epoch.append(temp_loss_per_epoch)
        print("Epoch : " + str(epoch+1) + " Completed")
        print("Time Taken for Epoch : " + str(ti) + " sec.")
        print("")
    
    print("Training the Network Completed...........")
    print("")
    
    print("Saving the model.........................")
    save_model(ss, config['model_file'], epoch)
    loss_per_epoch = np.array(loss_per_epoch)
    loss_per_epoch = np.true_divide(loss_per_epoch, all_images.shape[0])
    np.save(os.path.join(os.getcwd(), os.path.join(config['model_metrics'][0], config['model_metrics'][1])), (loss_per_epoch))
    print("Saving the model Completed...............")
    print("")
    ss.close()

def main():
    batch_train()

if __name__ == '__main__':
    main()

