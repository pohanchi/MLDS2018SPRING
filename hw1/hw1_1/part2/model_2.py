import tensorflow as tf 
import numpy as np 
import pickle


class FC5():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,784])
        self.y = tf.placeholder(tf.float64,[None,10])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
        # Build 5 Layers
        self.Layer_1 = tf.layers.dense(self.x,6,activation=tf.nn.relu,) 
        self.Layer_2 = tf.layers.dense(self.Layer_1,31,activation=tf.nn.relu,) 
        self.Layer_3 = tf.layers.dense(self.Layer_2,32,activation=tf.nn.relu,) 
        self.Layer_4 = tf.layers.dense(self.Layer_3,35,activation=tf.nn.relu,)
        self.Layer_5 = tf.layers.dense(self.Layer_4,30,activation=tf.nn.relu,) 
        self.Output  = tf.layers.dense(self.Layer_4,10,)  
        
        return 
    def loss_f(self):
        loss = tf.losses.softmax_cross_entropy(self.y, self.Output)
        self.loss = loss 
        self.loss_scalar = tf.summary.scalar('loss',self.loss)
        with tf.name_scope('train_loss'):
            self.loss_train = tf.summary.scalar('loss_train',self.loss)
        with tf.name_scope('test_loss'):
            self.loss_test = tf.summary.scalar('loss_test',self.loss)
    def optimizer_f(self):
        self.learning_rate = 0.005
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return

class FC3():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,784])
        self.y = tf.placeholder(tf.float64,[None,10])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
        # Build 3 Layers
        self.Layer_1 = tf.layers.dense(self.x,6,activation=tf.nn.relu,) 
        self.Layer_2 = tf.layers.dense(self.Layer_1,52,activation=tf.nn.relu,) 
        self.Layer_3 = tf.layers.dense(self.Layer_2,55,activation=tf.nn.relu,) 
        self.Output  = tf.layers.dense(self.Layer_3,10,)  
        
        return 
    def loss_f(self):
        loss = tf.losses.softmax_cross_entropy(self.y, self.Output)
        self.loss = loss 
        self.loss_scalar = tf.summary.scalar('loss',self.loss)
        with tf.name_scope('train_loss'):
            self.loss_train = tf.summary.scalar('loss_train',self.loss)
        with tf.name_scope('test_loss'):
            self.loss_test = tf.summary.scalar('loss_test',self.loss)
    def optimizer_f(self):
        self.learning_rate = 0.005
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return


class FC7():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,784])
        self.y = tf.placeholder(tf.float64,[None,10])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
        # Build 5 Layers
        self.Layer_1 = tf.layers.dense(self.x,6,activation=tf.nn.relu,) 
        self.Layer_2 = tf.layers.dense(self.Layer_1,15,activation=tf.nn.relu,) 
        self.Layer_3 = tf.layers.dense(self.Layer_2,24,activation=tf.nn.relu,) 
        self.Layer_4 = tf.layers.dense(self.Layer_3,35,activation=tf.nn.relu,)
        self.Layer_5 = tf.layers.dense(self.Layer_4,25,activation=tf.nn.relu,) 
        self.Layer_6 = tf.layers.dense(self.Layer_5,25,activation=tf.nn.relu,) 
        self.Layer_7 = tf.layers.dense(self.Layer_6,25,activation=tf.nn.relu,) 
        self.Output  = tf.layers.dense(self.Layer_7,10)  
        
        return 
    def loss_f(self):
        loss = tf.losses.softmax_cross_entropy(self.y, self.Output)
        self.loss = loss 
        self.loss_scalar = tf.summary.scalar('loss',self.loss)
        with tf.name_scope('train_loss'):
            self.loss_train = tf.summary.scalar('loss_train',self.loss)
        with tf.name_scope('test_loss'):
            self.loss_test = tf.summary.scalar('loss_test',self.loss)
    def optimizer_f(self):
        self.learning_rate = 0.005
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return


class CNN_1():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,784])
        self.y = tf.placeholder(tf.int32,[None,10])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        # self.accuracy_f()
        return 
    def model_structure(self):
        # Build 5 Layers
        input_layer = tf.reshape(self.x, [tf.shape(self.x)[0], 28, 28, 1])
        self.conv1 = tf.layers.conv2d(inputs=input_layer,filters=5,kernel_size=[10, 10],padding="same",activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[4, 4], strides=1)
        self.conv2 = tf.layers.conv2d(inputs=self.pool1,filters=2,kernel_size=[10, 10],padding="same",activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[3, 3], strides=1)
        self.pool2_flat = tf.reshape(self.pool2, [tf.shape(self.x)[0],1058]) 
        self.logits= tf.layers.dense(self.pool2_flat,10,activation=None)
        self.predictions = tf.argmax(input=self.logits, axis=1)
        
        return 
    def loss_f(self):
        loss = tf.losses.softmax_cross_entropy(self.y, self.logits)
        self.loss = loss 
        self.loss_scalar = tf.summary.scalar('loss',self.loss)

    def optimizer_f(self):
        self.learning_rate = 0.005
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return     
    # def accuracy_f(self):
    #     self.accuracy=tf.metrics.accuracy(labels=self.y, predictions=self.predictions["classes"])

class CNN_2():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,784])
        self.y = tf.placeholder(tf.int32,[None,10])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
        # Build 5 Layers
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        self.conv1 = tf.layers.conv2d(inputs=input_layer,filters=22,kernel_size=[2, 2],padding="same",activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=1)
        self.conv2 = tf.layers.conv2d(inputs=self.pool1,filters=4,kernel_size=[2, 2],padding="same",activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=1)
        self.conv3 = tf.layers.conv2d(inputs=self.pool2,filters=4,kernel_size=[2, 2],padding="same",activation=tf.nn.relu)
        self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=1)
        self.conv4 = tf.layers.conv2d(inputs=self.pool3,filters=2,kernel_size=[2, 2],padding="same",activation=tf.nn.relu)
        self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4, pool_size=[2, 2], strides=1)
        self.pool2_flat = tf.reshape(self.pool4, [tf.shape(self.x)[0],1152])
        self.logits= tf.layers.dense(self.pool2_flat,10)

        
        return 
    def loss_f(self):
        loss = tf.losses.softmax_cross_entropy(self.y, self.logits)
        self.loss = loss 
        self.loss_scalar = tf.summary.scalar('loss',self.loss)

    def optimizer_f(self):
        self.learning_rate = 0.005
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return     
    


class CNN_3():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,784])
        self.y = tf.placeholder(tf.int32,[None,10])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
        # Build 5 Layers
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        self.conv1 = tf.layers.conv2d(inputs=input_layer,filters=11,kernel_size=[2, 2],padding="same",activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=1)
        self.conv2 = tf.layers.conv2d(inputs=self.pool1,filters=8,kernel_size=[2, 2],padding="same",activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=1)
        self.conv3 = tf.layers.conv2d(inputs=self.pool2,filters=2,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
        self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[3, 3], strides=1)
        self.pool2_flat = tf.reshape(self.pool3, [tf.shape(self.x)[0],1152])
        self.logits= tf.layers.dense(self.pool2_flat,10)
        self.predictions = {"classes": tf.argmax(input=self.logits, axis=1),"probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")}

        
        return 
    def loss_f(self):
        loss = tf.losses.softmax_cross_entropy(self.y, self.logits)
        self.loss = loss 
        self.loss_scalar = tf.summary.scalar('loss',self.loss)

    def optimizer_f(self):
        self.learning_rate = 0.005
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return     
    

class CNN_4():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,784])
        self.y = tf.placeholder(tf.int32,[None,10])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
        # Build 5 Layers
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        self.conv1 = tf.layers.conv2d(inputs=input_layer,filters=4,kernel_size=[12, 12],padding="same",activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[12, 12], strides=1)
        self.pool2_flat = tf.reshape(self.pool1, [tf.shape(self.x)[0],1156])
        self.logits= tf.layers.dense(self.pool2_flat,10)

        
        return 
    def loss_f(self):
        loss = tf.losses.softmax_cross_entropy(self.y, self.logits)
        self.loss = loss 
        self.loss_scalar = tf.summary.scalar('loss',self.loss)

    def optimizer_f(self):
        self.learning_rate = 0.005
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return 