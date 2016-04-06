# coding=latin-1
#
# Last update : 05/04/2016
# Author : Naascraft
# Description : TensorFlow Tutorial : MNIST for ML Beginners
# Link : https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html

### Import module ###
from tensorflow.examples.tutorials.mnist import input_data
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
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    if _runTime: print("Total run time : " + str(time() - t) + " s.\n")
