# coding=latin-1
#
# Last update : 06/04/2016
# Author : Naascraft
# Description : TensorFlow Tutorial : Deep MNIST for Experts
# Link : https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html

### Import module ###
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import os
from time import time

### Command Line Arguments ###
_verb = "-v" in sys.argv
_help = "-h" in sys.argv
_runTime = "-rt" in sys.argv

### Functions ###
def weight_variable(shape):
    #Normal noise for weights init
    
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #Positive bias for init
    
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #Convolution definition
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #Pooling definition
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    if _runTime: t = time()
    
    ### Data importation ###
    ########################
    mnist = input_data.read_data_sets("../MnistDDL/MNIST_data/", one_hot=True)

    ### Variables and placeholders ###
    # Model description :            #
    #    Convolutional Network       #
    ##################################
    # Weights and bias
        # Layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
        # Layer 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
        # Fully-connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
        # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    # Real MNIST images and labels
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    # Reshaped images as 4d tensors
    x_image = tf.reshape(x, [-1,28,28,1])

    ### Layers definition ###
    #########################
    # First Convolutional Layer
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Second Convolutional Layer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # Densely Connected Layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Readout Layer
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    ### Training steps ###
    ######################
    # Cross-entropy as cost to minimize
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    # ADAM optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # Correct prediction
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ### Session launching ###
    #########################
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    ### Proper training ###
    #######################
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
            if _verb: print("step %d, training accuracy %g" % (i, train_accuracy))
        
        sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})

    ### Model evaluation ###
    ########################
    if _verb: print("Test accuracy : %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        
    
    if _runTime: print("Total run time : " + str(time() - t) + " s.\n")
