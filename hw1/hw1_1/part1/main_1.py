import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
from model import * 
import os
import pickle
import tqdm
np.seterr(divide='ignore', invalid='ignore')

#0,0486
def target_function1(x):
    # function should be sin(5* np.pi * x / (5* np.pi * x))
    y=np.sinc(5*x)
    return y 

if __name__ == '__main__':
    plt.style.use('bmh')
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    x   = np.linspace(-1.0, 1.0, num=16384)
    random.shuffle(x)
    y   = target_function1(x)
    data=np.stack((x,y),axis=1)

    epoch = 5000
    num_natch = int(16384  /128)
    batch_size = 128
    tmp =data
    
    learning_rates= [random.uniform(0.0005,0.0009),random.uniform(0.0001, 0.0005),random.uniform(0.001,0.005),random.uniform(0.005,0.009)]
    print('learning_rate = ',learning_rates)
    for k in tqdm.tqdm(learning_rates):
        tmp_try = []
        lr_tmp  = []
        FC_1=FC1()
        FC_1.learning_rate = k
        FC1_loss =FC_1.loss
        train_FC1=tf.train.AdamOptimizer(FC_1.learning_rate).minimize(FC1_loss)
        for try_ in range(3):
            data = tmp
            step = 0
            sess=tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            vars = tf.trainable_variables()
            vars_vals = sess.run(vars)
            loss_array = []
            step_array= []
            for i in range(epoch):
                # random.shuffle(data)
                for j in range(num_natch):
                    x_train=(data[j*batch_size:(j+1)*batch_size,0])[:,np.newaxis]
                    y_train=(data[j*batch_size:(j+1)*batch_size,1])[:,np.newaxis]
                    _ =sess.run(train_FC1,feed_dict={FC_1.x:x_train,FC_1.y:y_train})

                    if (((i+1) %2 == 0) & ((j+1)% 128==0)):
                        loss   = sess.run(FC1_loss,feed_dict={FC_1.x:x_train,FC_1.y:y_train})
                        if (((i+1) %1000 == 0) & ((j+1)% 128==0)):
                            print("epoch {} batch {} loss {}".format((i+1),(j+1),loss))

                        loss_array += [loss]
                        step_array+=[step]
                    step +=1
            sess.close()
            tmp_try += [loss_array]
            
            if try_ == 2:
                array=np.stack(tmp_try)
                mean=np.mean(array,axis = 0)
                lr_tmp = list(mean)
                plt.plot(step_array,lr_tmp,label="learning_rate={}".format(k))
                pickle.dump(lr_tmp,open('1_26_fc1_lr_{}.p'.format(k),'wb'))
    plt.yscale('symlog')
    plt.legend(loc='upper left')
    plt.title('FC1_LOSS')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig('1_28_FC1_Loss.png')
    plt.show()

            

                
            
                




