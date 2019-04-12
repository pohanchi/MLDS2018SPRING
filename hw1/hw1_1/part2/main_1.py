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

    data_list=list()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_img = mnist.train.images
    train_label = mnist.train.labels
    test_img = mnist.test.images
    test_label = mnist.test.labels
    index = list(range(len(train_img)))
    random.shuffle(index)

    epoch = 10
    batch_size = 100
    num_batch = int(55000 / 100)
    CNN = CNN_1()
    loss = CNN.loss
    learning_rates= [random.uniform(0.001, 0.005),random.uniform(0.001, 0.005),random.uniform(0.001, 0.005),random.uniform(0.00001, 0.00005),random.uniform(0.00001, 0.00005),random.uniform(0.00001, 0.00005),random.uniform(0.0001, 0.0005),random.uniform(0.0001, 0.0005),random.uniform(0.0001, 0.0005)]
    for learning_rate in learning_rates:
        
        CNN.learning_rate = learning_rate
        
        Train_step = tf.train.AdamOptimizer(CNN.learning_rate).minimize(loss)
        loss_array = []
        step = 0
        step_array = []
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(epoch):
            random.shuffle(index)
            train_img = train_img[index]
            train_label= train_label[index]
            for j in range(num_batch):
                x_data =train_img[j*100:(j+1)*100]
                y_data=train_label[j*100:(j+1)*100]
                _=sess.run(Train_step,feed_dict={CNN.x:x_data,CNN.y:y_data})
                if (j+1) % 55 == 0:
                    loss_show=sess.run(loss,feed_dict={CNN.x:x_data,CNN.y:y_data})
                    print("epoch %d num_batch %2d loss = %.5f" %(i,j,loss_show))
                    loss_array+=[loss_show]
                    step_array+=[step]
                step +=1
        plt.plot(step_array,loss_array,label='learning_rate= {}'.format(learning_rate))
        
        sess.close()
    plt.yscale('log')
    plt.title('CNN_1 Loss on training')
    plt.legend(loc='upper left')
    plt.style.use('ggplot')
    plt.savefig('two_hidden_layer_loss_step.png')
    plt.show()

        



    




        

