# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:03:22 2018

@author: meghs
"""
#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#DEFINE NECESSARY FUNCTIONS
def bsig(y):
    return 1/(1+np.exp(-y))
def d_bsig(fy):
    return fy*(1-fy)

#given data points 
A_0=np.array([[-1,-1,-1,-1,1,1,1,1],[-1,-1,1,1,-1,-1,1,1],[-1,1,-1,1,-1,1,-1,1]])
    
#A_0=np.array([[0,0,0,0,1,1,1,1],[0,0,1,1,0,0,1,1],[0,1,0,1,0,1,0,1]])
Y_g=np.array([[0,1,1,0,1,0,0,1]])
print('actual output needed')
print(Y_g)
data=8 #no. of data given
ip_units=3
h_units=2
op_units=1

#initialize weights and bias
np.random.seed(0)
W_1=np.random.uniform(low=-0.5 ,high=0.5,size=[h_units,ip_units]) 
b_1=np.random.uniform(size=[h_units,1]) 
b_1=np.tile(b_1,(1,data))
W_2=np.random.uniform(low=-0.5 ,high=0.5,size=[op_units,h_units]) 
b_2=np.random.uniform(size=[op_units,1])  
b_2=np.tile(b_2,(1,data))


for i in range(55):

    #forward pass
    Z_1=np.dot(W_1,A_0)+b_1 #input to hidden layer 
    A_1=bsig(Z_1) #output of hidden layer   
    Z_2=np.dot(W_2,A_1)+b_2        #input to output layer   
    A_2=bsig(Z_2) #output of output layer ie Y_o     
    Y_o=A_2    #1x8
    print('---------------------------calculated output---------------------')
    print(Y_o)
    error=(Y_g-Y_o)
   # print('-------------------------error------------------------')
    #print(error)
    alpha=0.1 #learning rate

    #backward pass
    #del_2=error*d_bsig(Z_2)
    del_2=Y_o-Y_g
    dW_2=alpha*np.dot(del_2,(A_1.T))
    db_2=alpha*del_2
    W_2=W_2-dW_2
    b_2=b_2-db_2
    
    del_1=W_2.T*del_2*d_bsig(Z_1)
    dW_1=alpha*np.dot(del_1,A_0.T)
    db_1=alpha*del_1
    W_1=W_1-dW_1
    b_1=b_1-db_1
    
    err = np.sum(error**2)*0.5
    
    print (err)
    plt.plot(i,err,'bo')
    
print('rounded output',np.round(Y_o))

# naming the x axis
plt.xlabel('no. of iterations')
# naming the y axis
plt.ylabel('error')
 
# giving a title to my graph
plt.title('iteration vs error')
plt.show()