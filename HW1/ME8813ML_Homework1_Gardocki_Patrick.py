#####################################################
##  ME8813ML Homework 1:
##  Implement a quasi-Newton optimization method for data fitting
#####################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search

########################################################
## Implement a parameter fitting function fit() so that
##  p = DFP_fit(x,y)
## returns a list of the parameters as p of model:
##  p[0] + p[1]*cos(2*pi*x) + p[2]*cos(4*pi*x) + p[3]*cos(6*pi*x) 
########################################################
# Fixing random state for reproducibility
np.random.seed(19680801)
dx = 0.1
x_lower_limit = 0
x_upper_limit = 40                                      
x = np.arange(x_lower_limit, x_upper_limit, dx)
data_size = len(x)                                 # data size
noise = np.random.randn(data_size)                 # white noise
# Original dataset
y = 2.0 + 3.0*np.cos(2*np.pi*x) + 1.0*np.cos(6*np.pi*x) + noise 
###########################################

def fun(p):
    ## Returns Minimization Function
    return sum((p[0] + p[1]*np.cos(2*np.pi*x) + p[2]*np.cos(4*np.pi*x) + p[3]*np.cos(6*np.pi*x) - y)**2)

def grad(p):
    ## Returns Gradient
    gradient = np.array(np.sum(np.vstack([
                2*(p[0] - y + p[2]*np.cos(4*np.pi*x) + p[3]*np.cos(6*np.pi*x) + p[1]*np.cos(2*np.pi*x)),
                2*np.cos(2*np.pi*x)*(np.cos(2*np.pi*x)*p[1] - y + p[2]*np.cos(4*np.pi*x) + p[3]*np.cos(6*np.pi*x) + p[0]),
                2*np.cos(4*np.pi*x)*(np.cos(4*np.pi*x)*p[2] - y + p[3]*np.cos(6*np.pi*x) + p[1]*np.cos(2*np.pi*x) + p[0]),
                2*np.cos(6*np.pi*x)*(np.cos(6*np.pi*x)*p[3] - y + p[2]*np.cos(4*np.pi*x) + p[1]*np.cos(2*np.pi*x) + p[0])]),
                axis=1))
    gradient = np.array(gradient)
    return gradient
 


def gradalpha(p):
    ## Returns Gradient for Line Search
    gradient = np.array([
                sum(2*(p[0] - y + p[2]*np.cos(4*np.pi*x) + p[3]*np.cos(6*np.pi*x) + p[1]*np.cos(2*np.pi*x))),
                sum(2*np.cos(2*np.pi*x)*(np.cos(2*np.pi*x)*p[1] - y + p[2]*np.cos(4*np.pi*x) + p[3]*np.cos(6*np.pi*x) + p[0])),
                sum(2*np.cos(4*np.pi*x)*(np.cos(4*np.pi*x)*p[2] - y + p[3]*np.cos(6*np.pi*x) + p[1]*np.cos(2*np.pi*x) + p[0])),
                sum(2*np.cos(6*np.pi*x)*(np.cos(6*np.pi*x)*p[3] - y + p[2]*np.cos(4*np.pi*x) + p[1]*np.cos(2*np.pi*x) + p[0]))],
                dtype=np.float32)
               
    return gradient


def get_dk(hess,grad):
    ## Returns Search Direction
    dk = -hess.dot(grad)
    return dk


def Hess(H,pk,qk):
    ## Returns Hessian Approximation
    p1 = H.dot(qk.reshape(-1,1))
    p2 = p1.dot(qk.reshape(-1,1).T)
    p3 = p2.dot(H)
    p4 = qk.reshape(-1,1).T.dot(H)
    p5 = p4.dot(qk)
    Hk = H + (pk.dot(pk.T))/(pk.T.dot(qk)) - p3/p5

    return Hk


def DFP_fit(x,y,eps):
    # Main DFP Method Algoritm
    pk11 = p0
    pk1 = np.array([1,1,10,10], dtype=np.float32)
    H11 = H
    while abs((fun(pk11) - fun(pk1))/fun(pk1)) > eps:
        pk1 = pk11
        gk = grad(pk11)
        dk = get_dk(H11,gk)
        alpha = line_search(fun,gradalpha,pk11,dk)
        pk11 = pk11 + alpha[0]*dk
        gk11 = grad(pk11)
        qk11 = gk11 - gk
        H11 = Hess(H,pk11,qk11)
        
    return pk11
 

H = np.identity(4) # Initial Hessian Guess
p0 = np.array([1,1,1,1], dtype=np.float32) # Initial Starting Point
eps = 0.00001 # Error Threshold 


p = DFP_fit(x,y,eps)
print(p)

y1 = p[0] + p[1]*np.cos(2*np.pi*x) + p[2]*np.cos(4*np.pi*x) + p[3]*np.cos(6*np.pi*x) + noise

###########################################
fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y)
axs[0].set_xlim(x_lower_limit, x_upper_limit)
axs[0].set_xlabel('x')
axs[0].set_ylabel('observation')
axs[0].grid(True)
#########################################
## Plot the predictions from your fitted model here
axs[1].plot(x,y1)
axs[1].set_xlim(x_lower_limit, x_upper_limit)
axs[1].set_xlabel('x')
axs[1].set_ylabel('model prediction')
fig.tight_layout()
plt.show()