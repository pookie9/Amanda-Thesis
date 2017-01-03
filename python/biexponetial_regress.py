import numpy as np
from math import exp
import random

def gradient_step(c1,c2,k1,k2,t,y,alpha):
#    print c1,c2,k1,k2
    c1_grad=-2*y*exp(-t*k1)-2*c1*exp(-2*t*k1)-2*c2*exp(-t*(k1+k2))
    c2_grad=-2*y*exp(-t*k2)-2*c2*exp(-2*t*k2)-2*c1*exp(-t*(k1+k2))
    k1_grad=2*y*c1*t*exp(-t*k1)+2*t*c1*exp(-2*t*k1)+2*t*c1*c2*exp(-t*(k1+k2))
    k2_grad=2*y*c2*t*exp(-t*k2)+2*t*c2*exp(-2*t*k2)+2*t*c1*c2*exp(-t*(k1+k2))
    c1-=c1_grad*alpha
    c2-=c2_grad*alpha
    c1=max(c1,0)
    c2=max(c2,0)
    s=c1+c2
    c1/=s
    c2/=s
    k1-=k1_grad*alpha
    k2-=k2_grad*alpha
    k1=max(k1,0.0)
    k2=max(k2,0.0)
    if c1+c2+.000000001<1.0:
        print "BAD HERE: ",c1,c2,s
    return (c1,c2,k1,k2)

def pred(c1,c2,k1,k2,t):
    return c1*exp(-t*k1)+c2*exp(-t*k2)

def mse(c1,c2,k1,k2,data):
    try:
        return sum([(pred(c1,c2,k1,k2,datum[0])-datum[1])**2 for datum in data])
    except:
        return float('inf')

def sgd(data):
    best_mse=float('inf')
    bestc1=0
    bestc2=0
    bestk1=0
    bestk2=0    
    for j in range(1000):
        c1=random.random()
        c2=1.0-c1
        k1=random.random()
        k2=random.random()*10
        prev_mse=float('inf')
        alpha=.01
        prevs=(0,0,0,0)
        for i in range(1000000):
            if i%1==0:
                cur_mse=mse(c1,c2,k1,k2,data)
                if cur_mse>prev_mse:
                    #c1,c2,k1,k2=prevs
                    break
                prev_mse=cur_mse
            t,y=random.choice(data)
            prevs=(c1,c2,k1,k2)

            c1,c2,k1,k2=gradient_step(c1,c2,k1,k2,t,y,alpha)
            alpha*=.9999
        if prev_mse<best_mse:
            c1,c2,k1,k2=prevs
            bestc1=c1
            bestc2=c2
            bestk1=k1
            bestk2=k2
            best_mse=prev_mse
    return (bestc1,bestc2,bestk1,bestk2,best_mse)

def run_default():
    random.seed(11)
    import sys
    filename='data/PLAPEG_VE822_intracranial.csv'
    if len(sys.argv)>1:
        filename=sys.argv[1]
    data=np.genfromtxt(filename,delimiter=',')[1:,:2]
    x=data[:,0]
    y=data[:,1]
    return sgd(zip(x,y))

if __name__=='__main__':
    print run_default()
