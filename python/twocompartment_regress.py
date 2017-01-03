import numpy as np
from math import exp
import random
LAMBDA=1.55
#LAMBDA=10000000
def gradient_step(c1,c2,k1,k2,t,y,alpha):
    p=pred(c1,c2,k1,k2,t)
    c1_grad=(p-y)*(LAMBDA*exp(-k1*t)-k1*exp(-LAMBDA*t))/(LAMBDA-k1)
    c2_grad=(p-y)*(LAMBDA*exp(-k2*t)-k2*exp(-LAMBDA*t))/(LAMBDA-k2)
    k1_grad=(y-p)*c1*((LAMBDA*t*exp(-k1*t)+exp(-LAMBDA*t))/(LAMBDA-k1)+(k1*exp(-LAMBDA*t)-LAMBDA*exp(-k1*t))/(LAMBDA-k1)**2.0)
    k2_grad=(y-p)*c2*((LAMBDA*t*exp(-k2*t)+exp(-LAMBDA*t))/(LAMBDA-k2)+(k2*exp(-LAMBDA*t)-LAMBDA*exp(-k2*t))/(LAMBDA-k2)**2.0)
    c1-=c1_grad*alpha
    c2-=c2_grad*alpha
    c1=max(c1,0)
    c2=max(c2,0)
    s=c1+c2
    c1/=s
    c2/=s
    k1-=k1_grad*alpha
    k2-=k2_grad*alpha
    if c1+c2+.000000001<1.0:
        print "BAD HERE: ",c1,c2,s
    return (c1,c2,k1,k2)

def pred(c1,c2,k1,k2,t):
    #return c1*exp(-t*k1)+c2*exp(-t*k2)
    return c1*(LAMBDA*exp(-k1*t)-k1*exp(-LAMBDA*t))/(LAMBDA-k1)+c2*(LAMBDA*exp(-k2*t)-k2*exp(-LAMBDA*t))/(LAMBDA-k2)

def mse(c1,c2,k1,k2,data):
    return sum([(pred(c1,c2,k1,k2,datum[0])-datum[1])**2 for datum in data])

def show_data(c1,c2,k1,k2,data):
    for datum in data:
        print datum[1],pred(c1,c2,k1,k2,datum[0])
def sgd(data):
    best_mse=float('inf')
    bestc1=0
    bestc2=0
    bestk1=0
    bestk2=0    
    for j in range(1000):
        c1=random.random()
        c2=1.0-c1
        k1=random.random()*1
        k2=random.random()*100
        prev_mse=float('inf')
        alpha=0.1
        prevs=(0,0,0,0)
        for iter in range(1000000):
            cur_mse=mse(c1,c2,k1,k2,data)
            if cur_mse>prev_mse:
                break
            prev_mse=cur_mse
            t,y=random.choice(data)
            prevs=(c1,c2,k1,k2)

            c1,c2,k1,k2=gradient_step(c1,c2,k1,k2,t,y,alpha)
            alpha*=.999
        if prev_mse<best_mse:
            c1,c2,k1,k2=prevs
            bestc1=c1
            bestc2=c2
            bestk1=k1
            bestk2=k2
            best_mse=prev_mse
    return bestc1,bestc2,bestk1,bestk2,best_mse


def run_default():    

    random.seed(10)  
    import sys
    filename='data/PLAPEG_VE822_intracranial.csv'
    if len(sys.argv)>1:
        filename=sys.argv[1]
    data=np.genfromtxt(filename,delimiter=',')[1:,:]
#    data=np.genfromtxt("data/synthetic.csv",delimiter=',')[0:,:]
    x=map(float,list(data[:,0]))
    y=map(float,list(np.average(data[:,1:],axis=1)))
    c1,c2,k1,k2,sse=sgd(zip(x,y))
    show_data(c1,c2,k1,k2,data)
    return (c1,c2,k1,k2,sse)


if __name__=='__main__':
    print run_default()
