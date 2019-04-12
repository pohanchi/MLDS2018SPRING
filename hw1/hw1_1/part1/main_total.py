
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
    y=np.sin(5* np.pi *x)/(5*x)
    return y 

if __name__ == '__main__':
    # plt.style.use('bmh')
    # plt.rcParams['font.family'] = 'monospace'
    # plt.rcParams['axes.labelweight'] = 'bold'

    x_1   = np.linspace(-1,1,16384)
    
    y   = target_function1(x_1)
    data = np.stack((x_1,y),axis=1)
    epoch = 25000
    num_natch = int(16384  /128)
    batch_size = 128

    tmp =data

    modes = [2,1,0]
    for mode in tqdm.tqdm(modes):
        tmp_try = []
        lr_tmp  = []
        tf.reset_default_graph()
        if mode == 0 :
            model=FC1()
            model_loss =model.loss
            model.learning_rate = 4e-4
            train_Step=tf.train.AdamOptimizer(model.learning_rate,).minimize(model_loss)
            name = 'FC1'
        if mode == 1:
            model=FC2()
            model_loss =model.loss
            model.learning_rate = 2e-3
            train_Step=tf.train.AdamOptimizer(model.learning_rate,).minimize(model_loss)
            name = 'FC2'
        if mode == 2:
            model=Shallow_Net()
            model_loss =model.loss
            model.learning_rate = 4e-3
            train_Step=tf.train.AdamOptimizer(model.learning_rate,).minimize(model_loss)
            name = 'Shallow'
        for try_ in range(1):
            data = tmp
            step = 1

            sess=tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            loss_array = []
            step_array= []
            for i in range(epoch):
                np.random.shuffle(data)
                for j in range(num_natch):
                    x_train=(data[j*batch_size:(j+1)*batch_size,0])[:,np.newaxis]
                    y_train=(data[j*batch_size:(j+1)*batch_size,1])[:,np.newaxis]
                    _ = sess.run(train_Step,feed_dict={model.x:x_train,model.y:y_train})
                    if (((i+1) %5 == 0) & ((j+1)% 128==0)):
                        loss   = sess.run(model_loss,feed_dict={model.x:x_train,model.y:y_train})
                        loss_array += [loss]
                        step_array+=[step]
                        if (((i+1) %250 == 0) & ((j+1)% 128==0)):
                            print("epoch {} batch {} loss {}".format((i+1),(j+1),loss))
                    step +=1
            
            sess.close()
            tmp_try += [loss_array]

            if try_ == 0:
                array=np.stack(tmp_try)
                mean=np.mean(array,axis = 0)
                lr_tmp = list(mean)
                plt.plot(step_array,lr_tmp,label=name)
                pickle.dump(lr_tmp,open(name+'.p','wb'))
                
    plt.yscale('symlog')
    plt.legend(loc='upper left')
    plt.title('Differences deeper vs Loss')
    plt.xticks(list(range(0,128*epoch,128)),list(range(epoch)))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('detail_Differences_on_deeper_vs_Loss.png')
    plt.show()
