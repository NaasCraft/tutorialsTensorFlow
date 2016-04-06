# coding=latin-1
#
# Last update : 05/04/2016
# Author : Naascraft
# Description : TensorFlow Tutorial : MNIST for ML Beginners
# Link : https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html

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

if __name__ == "__main__":
    if _runTime: t = time()
    
    ### Data importation ###
    ########################
    mnist = input_data.read_data_sets("../MnistDDL/MNIST_data/", one_hot=True)

    ### Variables and placeholders ###
    # Model description :            #
    #       y = softmax(W.x + b)     #
    ##################################
    # Any number of MNIST images (784-dimensional vector)
    x = tf.placeholder(tf.float32, [None, 784])

    # Weights matrix and bias vector
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Correct labels (one-hot vectors)
    y_ = tf.placeholder(tf.float32, [None, 10])

    ### Model training definition ###
    #################################
    # Softmax model definition
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Cross-entropy
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # Optimization algorithm
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    ### Session launching ###
    #########################
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    ### Proper training ###
    #######################
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    ### Model evaluation ###
    ########################
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    if _verb: print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
    if _runTime: print("Total run time : " + str(time() - t) + " s.\n")
        
    ### Session closing ###
    #######################
    sess.close()
