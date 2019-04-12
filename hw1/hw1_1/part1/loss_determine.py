import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pickle

if __name__ == '__main__':
    names=os.listdir('./')
    for files in names:
        if files[:6] == "fc1_lr":
            array=pickle.load(open(files,'rb'))
            x_axis = range(len(array))
            plt.plot(x_axis,array,label='learning_rate_{}'.format(files[7:-3]))
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.show()
