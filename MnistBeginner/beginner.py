# coding=latin-1
#
# Last update : 05/04/2016
# Author : Naascraft
# Description : TensorFlow Tutorial : MNIST for ML Beginners
# Link : https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html

### Import module ###
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

### Command Line Arguments ###
_verb = "-v" in sys.argv
_help = "-h" in sys.argv

### Path variables ###
fileDir_ = os.path.dirname(os.path.realpath('__file__'))
dataPath_ = os.path.join( fileDir_, "../source/data/")
picklePath_ = os.path.join( fileDir_, "pickles/" )
outPath_ = os.path.join( fileDir_, "submission/tmp/")
modelPath_ = os.path.join( fileDir_, "models/tmp/")


if __name__ == "__main__":
    ### Data importation ###
    ########################
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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
        batch_xxs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    ### Model evaluation ###
    ########################
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(coreect_prediction, tf.float32))
    
    if verb_: print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))