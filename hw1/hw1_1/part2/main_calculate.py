import numpy as np 
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import random

from model_2 import *

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def calculate():
    vars = tf.trainable_variables()
    for var in vars:
        print(var)
    all_number=sum([np.prod(var.get_shape()) for var in vars])
    print('you use %d parameters' %(all_number))
    return
if __name__ == '__main__':
    
    # data_list=list()
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # train_img = mnist.train.images
    # train_label = mnist.train.labels
    # test_img = mnist.test.images
    # test_label = mnist.test.labels
    # index = np.array(range(len(train_img)))
    # random.shuffle(index)

    # epoch = 5
    # batch_size = 100
    # num_batch = int(55000 / 100)
    FCN = FC3()
    loss = FCN.loss
    learning_rate = 0.001
    FCN.learning_rate = learning_rate
    
    Train_step = tf.train.AdamOptimizer(FCN.learning_rate).minimize(loss)
    
    # loss_array = []
    step = 0
    step_array = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    calculate()
    print("This is fully connected 3 layers weight number")
    print("==================================================")
    sess.close()
    
    tf.reset_default_graph()
    FCN=FC5()
    loss = FCN.loss
    learning_rate = 0.001
    FCN.learning_rate = learning_rate
    Train_step = tf.train.AdamOptimizer(FCN.learning_rate).minimize(loss)
    sess= tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    calculate()
    print("This is fully connected 5 layers weight number")
    print("==================================================")
    sess.close()
    
    tf.reset_default_graph()
    sess = tf.Session()
    FCN=FC7()
    loss = FCN.loss
    learning_rate = 0.001
    FCN.learning_rate = learning_rate
    Train_step = tf.train.AdamOptimizer(FCN.learning_rate).minimize(loss)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    calculate()
    print("This is fully connected 7 layers weight number")
    print("==================================================")


