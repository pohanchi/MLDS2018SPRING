import tensorflow as tf 

class FC1():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,1])
        self.y = tf.placeholder(tf.float64,[None,1])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
        self.C = 0.1
        # Build 10 Layers
        self.Layer_1 = tf.layers.dense(self.x,5,activation=tf.nn.relu,)# 6
        self.Layer_2 = tf.layers.dense(self.Layer_1,10,activation=tf.nn.relu,) # 60
        self.Layer_3 = tf.layers.dense(self.Layer_2,10,activation=tf.nn.relu,) # 110
        self.Layer_4 = tf.layers.dense(self.Layer_3,5,activation=tf.nn.relu,)  # 55
        self.Output  = tf.layers.dense(self.Layer_4,1)  #6
        #Total Parameter numbers  237
        return 
    def loss_f(self):
        loss = tf.losses.mean_squared_error(self.y, self.Output) 
        self.loss = loss 
        
    def optimizer_f(self):
        self.learning_rate = 0.005
        
        return




class FC2():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,1])
        self.y = tf.placeholder(tf.float64,[None,1])
        self.model_structre()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structre(self):
        # Build 6 Layers
        self.C = 0.1
        self.Layer1 = tf.layers.dense(self.x,20,activation=tf.nn.relu,) #40
        self.Layer2 = tf.layers.dense(self.Layer1,9,activation=tf.nn.relu,)   #189
        self.Output = tf.layers.dense(self.Layer2,1)    #9
        #Total Parameter numbers 238  
        return
    def loss_f(self):
        loss = tf.losses.mean_squared_error(self.y, self.Output) 
        self.loss = loss 
    def optimizer_f(self):
        self.learning_rate = 0.005
        return


class Shallow_Net():
    def __init__(self):
        self.x = tf.placeholder(tf.float64,[None,1])
        self.y = tf.placeholder(tf.float64,[None,1])
        self.model_structure()
        self.loss_f()
        self.optimizer_f()
        return 
    def model_structure(self):
    
        self.Layer = tf.layers.dense(self.x,79,activation=tf.nn.relu,) # 158
        self.Output= tf.layers.dense(self.Layer, 1) # 79
        #Total Parameter numbers 237
        return 
    def loss_f(self):
        loss = tf.losses.mean_squared_error(self.y, self.Output) 
        self.loss = loss 
        
    def optimizer_f(self):
        self.learning_rate = 0.005
        return




