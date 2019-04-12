import tensorflow as tf 
import matplotlib.pyplot as plt
import random
from model import * 
import os
import pickle
import numpy as np
import tqdm

if __name__ == '__main__':
    modes =[2,1,0]
    for mode in modes:
        if mode == 0 :
            name= "One"
            x = list(np.linspace(-1,1,30))
            y = list(np.linspace(5,8,30))
        if mode == 1 :
            name= "two"
            x = list(np.linspace(12,13,30))
            y = list(np.linspace(19,22,30))
        if mode ==2:
            name= "three"
            x = list(np.linspace(15,17,30))
            y = list(np.linspace(30,33,30))
        plt.plot(x,y,label=name)

    plt.yscale('linear')
    plt.legend(loc='upper left')
    plt.show()