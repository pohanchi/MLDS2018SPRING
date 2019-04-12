from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import random
import tqdm

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
    
    epoch = 30
    batch_size = 100
    num_batch = int(55000 / 100)
    modes = [3,1,2,0]
    for mode in tqdm.tqdm(modes):
        tf.reset_default_graph()
        if mode == 0:
            name = "two_layers_cnn"
            CNN = CNN_1()
            loss = CNN.loss
            learning_rate = 0.0001
        if mode == 1:
            name = "four_layers_cnn"
            CNN = CNN_2()
            loss = CNN.loss
            learning_rate = 0.0002
        if mode == 2:
            name = "three_layers_cnn"
            CNN = CNN_3()
            loss = CNN.loss
            learning_rate = 0.0003
        if mode == 3:
            name = "one_layers_cnn"
            CNN = CNN_4()
            loss = CNN.loss
            learning_rate = 0.0001
        CNN.learning_rate = learning_rate
        Train_step = tf.train.AdamOptimizer(CNN.learning_rate).minimize(loss)
        loss_array = []
        step = 1
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
                if (j+1) % 11 == 0:
                    loss_show=sess.run(loss,feed_dict={CNN.x:x_data,CNN.y:y_data})
                    loss_array+=[loss_show]
                    step_array+=[step]
                    if (i+1) % 10 == 0:
                        print("epoch %d num_batch %2d loss = %.5f" %((i+1),(j+1),loss_show))
                step +=1
        plt.plot(step_array,loss_array,label=name)
    
        sess.close()
    plt.yscale('symlog')
    plt.title('Loss vs difference layers')
    plt.legend(loc='upper left')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.style.use('bmh')
    plt.xticks(list(range(0,550*epoch,550)),list(range(epoch)))
    plt.savefig('Detail_Loss_vs_difference_layers_cnn.png')
    plt.show()
