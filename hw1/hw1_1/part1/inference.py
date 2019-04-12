import tensorflow as tf 
import numpy as np 
from model import *
import matplotlib.pyplot as plt 

def mode_change(mode):
    if mode == 0:
        model_s = FC1()
        model_name = 'fc1_model/fc1.ckpt'
    if mode == 1:
        model_s =FC2()
        model_name = 'fc2_model/fc2.ckpt'
    if mode == 2:
        model_s =Shallow_Net()
        model_name = "Shallow_model/shallow.ckpt"
    return model_s,model_name 

def target_function1(x):
    # function should be sin(5* np.pi * x / (5* np.pi * x))
    y=np.sin(5* np.pi *x)/(5*np.pi*x)
    return y 

def target_function2(x):
    #function should be sgn(sin(5*np.pi * x))
    y=np.sign(np.sin(5 * np.pi * x))
    return y

if __name__ == "__main__":
    x   = np.linspace(-1,1,8192)[:,np.newaxis]
    # random.shuffle(x)
    y   = target_function1(x)
    plt.plot(x,y,color='blue',label='Ground_Truth')
    
    model_s, model_name =mode_change(2)
    # model_s, model_name =mode_change(2)
    saver = tf.train.Saver()

    # with tf.Session() as sess:
        
    #     saver.restore(sess,model_name)
    #     predict=sess.run(model_s.Output,feed_dict={model_s.x:x,model_s.y:y})
    
    # plt.plot(x,predict,color='red',label='FC_1')

    # model_s, model_name =mode_change(1)
    # with tf.Session() as sess:
    #     saver.restore(sess,model_name)
    #     predict=sess.run(model_s.Output,feed_dict={model_s.x:x,model_s.y:y})

    # plt.plot(x,predict,color='orange',label='FC_2')

    # model_s, model_name =mode_change(2)
    with tf.Session() as sess:
        saver.restore(sess,model_name)
        predict=sess.run(model_s.Output,feed_dict={model_s.x:x,model_s.y:y})
    plt.plot(x,predict,color='green',label='Shallow')

    plt.legend(loc='upper left')
    plt.show()
