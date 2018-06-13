# @author : Abhishek R S

import os
import numpy as np
import tensorflow as tf

class Network_Architecture:

    # initialize network parameters
    def __init__(self, img_pl, kernel_size, num_kernels, strides, data_format, padding, pool_size, training_pl, neurons, num_classes, dropout_rate = 0.5, reduction_strides = None):
        self.img_pl = img_pl
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.strides = strides
        self.data_format = data_format
        self.padding = padding 
        self.pool_size = pool_size
        self.training = training_pl
        self.neurons = neurons
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.reduction_strides = reduction_strides
        self.avg_pool_axes = None
        self.bn_axis = None
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1)

        if self.data_format == "channels_last":
            self.avg_pool_axes = [1, 2]
            self.bn_axis = -1
        else:          
            self.avg_pool_axes = [2, 3]
            self.bn_axis = 1


    # build a network based on vgg encoder
    def vgg_encoder(self):
        # encoder 1
        self.conv1_1 = self._get_conv2d_layer(self.img_pl, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, "conv1_1")
        self.bn1_1 = self._get_batch_norm_layer(self.conv1_1, "batch_norm1_1")
        self.elu1_1 = self._get_elu_activation(self.bn1_1, "elu1_1") 
        self.conv1_2 = self._get_conv2d_layer(self.elu1_1, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, "conv1_2") 
        self.bn1_2 = self._get_batch_norm_layer(self.conv1_2, "batch_norm1_2")
        self.elu1_2 = self._get_elu_activation(self.bn1_2, "elu1_2")
        self.pool1 = self._get_maxpool2d_layer(self.elu1_2, self.pool_size, self.pool_size, self.padding, self.data_format, "pool1") 

        # encoder 2
        self.conv2_1 = self._get_conv2d_layer(self.pool1, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, "conv2_1") 
        self.bn2_1 = self._get_batch_norm_layer(self.conv2_1, "batch_norm2_1")
        self.elu2_1 = self._get_elu_activation(self.bn2_1, "elu2_1") 
        self.conv2_2 = self._get_conv2d_layer(self.elu2_1, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, "conv2_2") 
        self.bn2_2 = self._get_batch_norm_layer(self.conv2_2, "batch_norm2_2")
        self.elu2_2 = self._get_elu_activation(self.bn2_2, "elu2_2") 
        self.pool2 = self._get_maxpool2d_layer(self.elu2_2, self.pool_size, self.pool_size, self.padding, self.data_format, "pool2") 

        # flatten and add fully connected layer with dropout
        self.flatten = self._get_flattened_features(self.pool2, "flatten")
        self.dense1 = self._get_dense_layer(self.flatten, self.neurons[0], "dense1")
        self.dropout = self._get_dropout_layer(self.dense1, self.dropout_rate, self.training, "dropout")
        self.elu_dense = self._get_elu_activation(self.dropout, "elu_dense")

        self.logits = self._get_dense_layer(self.elu_dense, self.num_classes, "logits")


    # build a network based on normal residual encoder
    def residual_encoder(self):
        # encoder 1
        self.elu1_0 = self._strided_convolution_block(self.img_pl, self.num_kernels[0], self.kernel_size, self.strides, self.padding, self.data_format, 1, 0)

        # encoder 2
        self.elu2_0 = self._strided_convolution_block(self.elu1_0, self.num_kernels[1], self.kernel_size, self.reduction_strides, self.padding, self.data_format, 2, 0)
        self.elu2_1 = self._strided_convolution_block(self.elu2_0, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 1)
        self.fuse2_1 = self._residual_block(self.elu2_1, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 2)
        self.elu2_3 = self._get_elu_activation(self.fuse2_1, "elu2_3")
        self.fuse2_2 = self._residual_block(self.elu2_3, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 4)
        self.elu2_5 = self._get_elu_activation(self.fuse2_2, "elu2_5")
        self.fuse2_3 = self._residual_block(self.elu2_5, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 6)
        self.elu2_7 = self._get_elu_activation(self.fuse2_3, "elu2_7")

        # encoder 3
        self.elu3_0 = self._strided_convolution_block(self.elu2_7, self.num_kernels[2], self.kernel_size, self.reduction_strides, self.padding, self.data_format, 3, 0)
        self.elu3_1 = self._strided_convolution_block(self.elu3_0, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 1)
        self.fuse3_1 = self._residual_block(self.elu3_1, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 2)
        self.elu3_3 = self._get_elu_activation(self.fuse3_1, "elu3_3")
        self.fuse3_2 = self._residual_block(self.elu3_3, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 4)
        self.elu3_5 = self._get_elu_activation(self.fuse3_2, "elu3_5")
        self.fuse3_3 = self._residual_block(self.elu3_5, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 6)
        self.elu3_7 = self._get_elu_activation(self.fuse3_3, "elu3_7")

        self.avg_pool = self._avg_pool(self.elu3_7, axis = self.avg_pool_axes, name = "avg_pool")
        self.logits = self._get_dense_layer(self.avg_pool, self.num_classes, "logits")  


    # build a network based on pre-activation residual encoder
    def preactivation_residual_encoder(self):
        # encoder 1
        self.conv1_0 = self._get_conv2d_layer(self.img_pl, self.num_kernels[0], self.kernel_size, self.strides, self.padding, self.data_format, "conv1_0")
        self.bn1_0 = self._get_batch_norm_layer(self.conv1_0, "batch_norm1_0")
        self.elu1_0 = self._get_elu_activation(self.bn1_0, "elu1_0")

        # encoder 2
        self.conv2_0 = self._get_conv2d_layer(self.elu1_0, self.num_kernels[1], self.kernel_size, self.reduction_strides, self.padding, self.data_format, "conv2_0")
        self.bn2_0 = self._get_batch_norm_layer(self.conv2_0, "batch_norm2_0")
        self.elu2_0 = self._get_elu_activation(self.bn2_0, "elu2_0")
        self.conv2_1 = self._get_conv2d_layer(self.elu2_0, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, "conv2_1")
        self.fuse2_1 = self._preactivation_residual_block(self.conv2_1, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 1)
        self.fuse2_2 = self._preactivation_residual_block(self.fuse2_1, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 3)
        self.fuse2_3 = self._preactivation_residual_block(self.fuse2_2, self.num_kernels[1], self.kernel_size, self.strides, self.padding, self.data_format, 2, 5)
    
        # encoder 3
        self.conv3_0 = self._get_conv2d_layer(self.fuse2_3, self.num_kernels[2], self.kernel_size, self.reduction_strides, self.padding, self.data_format, "conv3_0")
        self.bn3_0 = self._get_batch_norm_layer(self.conv3_0, "batch_norm3_0")
        self.elu3_0 = self._get_elu_activation(self.bn3_0, "elu3_0")
        self.conv3_1 = self._get_conv2d_layer(self.elu3_0, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, "conv3_1")
        self.fuse3_1 = self._preactivation_residual_block(self.conv3_1, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 1)
        self.fuse3_2 = self._preactivation_residual_block(self.fuse3_1, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 3)
        self.fuse3_3 = self._preactivation_residual_block(self.fuse3_2, self.num_kernels[2], self.kernel_size, self.strides, self.padding, self.data_format, 3, 5)

        self.avg_pool = self._avg_pool(self.fuse3_3, axis = self.avg_pool_axes, name = "avg_pool")
        self.logits = self._get_dense_layer(self.avg_pool, self.num_classes, "logits")  


    # build a strided convolution block
    def _strided_convolution_block(self, input_layer, num_kernels, kernel_size, strides, padding, data_format, num_1, num_2):
        _conv1 = self._get_conv2d_layer(input_layer, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num_1) + "_" + str(num_2))
        _bn1 = self._get_batch_norm_layer(_conv1, "batch_norm" + str(num_1) + "_" + str(num_2))
        _elu1 = self._get_elu_activation(_bn1, "elu" + str(num_1) + "_" + str(num_2))
        return _elu1


    # build a normal residual block
    def _residual_block(self, input_layer, num_kernels, kernel_size, strides, padding, data_format, num_1, num_2):
        _conv1 = self._get_conv2d_layer(input_layer, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num_1) + "_" + str(num_2))
        _bn1 = self._get_batch_norm_layer(_conv1, "batch_norm" + str(num_1) + "_" + str(num_2))
        _elu1 = self._get_elu_activation(_bn1, "elu" + str(num_1) + "_" + str(num_2))
        _conv2 = self._get_conv2d_layer(_elu1, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num_1) + "_" + str(num_2 + 1))
        _bn2 = self._get_batch_norm_layer(_conv2, "batch_norm" + str(num_1) + "_" + str(num_2 + 1))
        _fuse = tf.add(input_layer, _bn2, "fuse" + str(num_1) + "_" + str(num_2))
        return _fuse 


    # build a pre activation residual block
    def _preactivation_residual_block(self, input_layer, num_kernels, kernel_size, strides, padding, data_format, num_1, num_2):
        _bn1 = self._get_batch_norm_layer(input_layer, "batch_norm" + str(num_1) + "_" + str(num_2))
        _elu1 = self._get_elu_activation(_bn1, "elu" + str(num_1) + "_" + str(num_2))
        _conv1 = self._get_conv2d_layer(_elu1, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num_1) + "_" + str(num_2 + 1))
        _bn2 = self._get_batch_norm_layer(_conv1, "batch_norm" + str(num_1) + "_" + str(num_2 + 1))
        _elu2 = self._get_elu_activation(_bn2, "elu" + str(num_1) + "_" + str(num_2 + 1))
        _conv2 = self._get_conv2d_layer(_elu2, num_kernels, kernel_size, strides, padding, data_format, "conv" + str(num_1) + "_" + str(num_2 + 2))
        _fuse = tf.add(input_layer, _conv2, "fuse" + str(num_1) + "_" + str(num_2))
        return _fuse 


    # perform avg pool
    def _avg_pool(self, input_layer, axis, name = "avg_pool"): 
        return tf.reduce_mean(input_layer, axis = axis, name = name)


    # return Convolution2D layer
    def _get_conv2d_layer(self, input_tensor, num_filters, kernel_size, strides, padding, data_format, name = "conv"):
        return tf.layers.conv2d(inputs = input_tensor, filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format, kernel_regularizer = self.l2_regularizer, name = name)


    # return ELU activation function
    def _get_elu_activation(self, input_tensor, name = "elu"):
        return tf.nn.elu(input_tensor, name = name)
   

    # return batch normalization layer
    def _get_batch_norm_layer(self, input_tensor, name = "bn"):
        return tf.layers.batch_normalization(input_tensor, axis = self.bn_axis, training = self.training, name = name)


    # return MaxPool2D layer
    def _get_maxpool2d_layer(self, input_tensor, pool_size, strides, padding, data_format, name = "pool"):
        return tf.layers.max_pooling2d(inputs = input_tensor, pool_size = pool_size, strides = strides, padding = padding, data_format = data_format, name = name)


    # return the dense layer
    def _get_dense_layer(self, input_tensor, num_neurons, name = "dense"):
        return tf.layers.dense(input_tensor, units = num_neurons, name = name)


    # return the flattened features
    def _get_flattened_features(self, input_tensor, name = "flatten"):
        return tf.layers.flatten(input_tensor, name = name)


    # return the dropout layer
    def _get_dropout_layer(self, input_tensor, rate = 0.5, training = False, name = "dropout"):
        return tf.layers.dropout(inputs = input_tensor, rate = rate, training = training, name = name)

