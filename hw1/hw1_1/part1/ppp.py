import random 
import numpy as np

a=list(range(10))
b=list(range(10,20))
c=np.stack((a,b),axis=1)
# for i in range(len(a)):
#     c += [(a[i],b[i])]
for _ in range(10):
    
    random.shuffle(c)
    print(c[:10])
    print("===========")

    random.shuffle(c)
    print(c[:10])
    print("===========")