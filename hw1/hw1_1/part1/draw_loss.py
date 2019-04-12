import matplotlib.pyplot as plt 
import numpy as np 
import pickle

f1_name = 'FC1.p'
f2_name = 'FC2.p'
shallow_name = 'Shallow.p'

print(plt.style.available)
plt.style.use('ggplot')
log_f1 = pickle.load(open(f1_name,'rb'))
log_f2 = pickle.load(open(f2_name,'rb'))

log_shallow = pickle.load(open(shallow_name,'rb'))
step = 200000
x_array=[]
loss_array =[]

x_axis = range(0,128*step,6400)
loss_array = log_f1
plt.plot(x_axis,loss_array,color='blue',label='fc1')
x_array=[]
loss_array =[]

x_axis = range(0,128*step,6400)
loss_array = log_f2

plt.plot(x_axis,loss_array,color='red',label='fc2')
x_array=[]
loss_array =[]

x_axis = range(0,128*step,6400)
loss_array = log_shallow
epoch = 200000
plt.plot(x_axis,loss_array,color='orange',label='shallow')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(list(range(0,128*epoch,1280000)),list(range(0,epoch,10000)))
plt.yscale('log')
plt.legend(loc='upper right')
plt.show()


