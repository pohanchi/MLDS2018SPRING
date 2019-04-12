
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
from model import * 
import os
import pickle
import numpy as np
import tqdm
np.seterr(divide='ignore', invalid='ignore')


def target_function1(x):
    # function should be sin(5* np.pi * x / (5* np.pi * x))
    y=np.sinc(5*x)
    return y 

if __name__ == '__main__':
    # plt.style.use('bmh')
    # plt.rcParams['font.family'] = 'monospace'
    # plt.rcParams['axes.labelweight'] = 'bold'

    x   = np.linspace(-1.0, 1.0, num=16384)
    print(x[:5])
    random.shuffle(x)
    y   = target_function1(x)
    print(y[:5])
    data=np.stack((x,y),axis=1)
    print(data[:5])

    Shallow=Shallow_Net()
    Shallow_loss =Shallow.loss

    epoch = 5000
    num_natch = int(16384  /128)
    batch_size = 128

    tmp =data
    x_test= np.linspace(-1,1,128)[:,np.newaxis]
    y_test= target_function1(x_test)
    learning_rates= [random.uniform(0.0005, 0.0009),random.uniform(0.0001, 0.0005),random.uniform(0.005, 0.009),random.uniform(0.001,0.005)]
    print('learning_rate = ',learning_rates)
    for k in tqdm.tqdm(learning_rates):
        tmp_try = []
        lr_tmp  = []
        Shallow.learning_rate = k
        print(Shallow.learning_rate)
        train_Shallow=tf.train.AdamOptimizer(Shallow.learning_rate).minimize(Shallow_loss)
        for try_ in range(4):
            data = tmp
            step = 0
            
            sess=tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            loss_array = []
            step_array= []
            for i in range(epoch):
                # random.shuffle(data)
                for j in range(num_natch):
                    x_train=(data[j*batch_size:(j+1)*batch_size,0])[:,np.newaxis]
                    # print(x_train[:5])
                    y_train=(data[j*batch_size:(j+1)*batch_size,1])[:,np.newaxis]
                    # print(y_train[:5])
                    # print("===================")
                    _ = sess.run(train_Shallow,feed_dict={Shallow.x:x_train,Shallow.y:y_train})
                    if (((i+1) %2 == 0) & ((j+1)% 128==0)):
                        loss   = sess.run(Shallow_loss,feed_dict={Shallow.x:x_train,Shallow.y:y_train})
                        if (((i+1) %1000 == 0) & ((j+1)% 128==0)):
                            print("epoch {} batch {} loss {}".format((i+1),(j+1),loss))
                        loss_array += [loss]
                        step_array+=[step]
                    step +=1
            
            sess.close()
            
            tmp_try += [loss_array]

            if try_ == 3:
                array=np.stack(tmp_try)
                print(array.shape)
                mean=np.mean(array,axis = 0)
                print(mean.shape)
                lr_tmp = list(mean)
                plt.plot(step_array,lr_tmp,label="learning_rate={}".format(k))
                pickle.dump(lr_tmp,open('1_26_shallow_lr_{}.p'.format(k),'wb'))
    plt.yscale('symlog')
    plt.legend(loc='upper left')
    plt.title('Shallow_LOSS')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig('1_26_shallow_Loss.png')
    plt.show()
