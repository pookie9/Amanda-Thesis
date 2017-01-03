import numpy as np
from math import exp
import random
LAMBDA=1.55
#LAMBDA=10000000
def gradient_step(c1,c2,k1,k2,k3,t,y,alpha):
    p=pred(c1,c2,k1,k2,k3,t)
    c1_grad=(p-y)*((LAMBDA-k3)*exp(-t*(k1+k3))-k1*exp(-LAMBDA*t))/(LAMBDA-(k1+k3))
    c2_grad=(p-y)*((LAMBDA-k3)*exp(-t*(k2+k3))-k2*exp(-LAMBDA*t))/(LAMBDA-(k2+k3))
    k1_grad=(y-p)*c1*((exp(-LAMBDA*t)+t*(LAMBDA-k3)*exp(-t*(k1+k3)))/(LAMBDA-(k1+k3))+(k1*exp(-LAMBDA*t)-(LAMBDA-k3)*exp(-t*(k1+k3)))/(LAMBDA-(k1+k3))**2)
    k2_grad=(y-p)*c2*((exp(-LAMBDA*t)+t*(LAMBDA-k3)*exp(-t*(k2+k3)))/(LAMBDA-(k2+k3))+(k2*exp(-LAMBDA*t)-(LAMBDA-k3)*exp(-t*(k2+k3)))/(LAMBDA-(k2+k3))**2)
    k3_grad=(p-y)*(c1*((k3*t-LAMBDA*t-1)*exp(-t*(k1+k3))/(LAMBDA-(k1+k3))+((LAMBDA-k3)*exp(-t*(k1+k3))-k1*c1*exp(-LAMBDA*t))/(LAMBDA-k1-k3)**2)+c2*((k3*t-LAMBDA*t-1)*exp(-t*(k2+k3))/(LAMBDA-(k2+k3))+((LAMBDA-k3)*exp(-t*(k2+k3))-k2*c2*exp(-LAMBDA*t))/(LAMBDA-k2-k3)**2)) #CHECK FOR ERROR IN HERE!
    c1-=c1_grad*alpha
    c2-=c2_grad*alpha
    c1=max(c1,0)
    c2=max(c2,0)
    s=c1+c2
    c1/=s
    c2/=s
    k1-=k1_grad*alpha
    k2-=k2_grad*alpha
    k3-=k3_grad*alpha
    k1=max(k1,0.0)
    k2=max(k2,0.0)
    k3=max(k3,0.0)
    if c1+c2+.000000001<1.0:
        print "BAD HERE: ",c1,c2,s
    return (c1,c2,k1,k2,k3)

def pred(c1,c2,k1,k2,k3,t):
    return (c1*(LAMBDA-k3)*exp(-t*(k1+k3))-k1*c1*exp(-LAMBDA*t))/(LAMBDA-(k1+k3))+(c2*(LAMBDA-k3)*exp(-t*(k2+k3))-k2*c2*exp(-LAMBDA*t))/(LAMBDA-(k2+k3))

def show_data(c1,c2,k1,k2,k3,data):
    for datum in data:
        print datum[1],pred(c1,c2,k1,k2,k3,datum[0])

def mse(c1,c2,k1,k2,k3,data):
    try:
        return sum([(pred(c1,c2,k1,k2,k3,datum[0])-datum[1])**2 for datum in data])
    except:
        return float('inf')

def sgd(data):
    best_mse=float('inf')
    bestc1=0
    bestc2=0
    bestk1=0
    bestk2=0
    bestk3=0
    num_iters=1000
    import sys
    for j in range(num_iters):
        sys.stdout.write("\r" + str(j*100.0/num_iters)+"% BEST SSE: "+str(best_mse))
        sys.stdout.flush()
        c1=random.random()
        c2=1.0-c1
        k1=random.random()
        k2=random.random()*100
        k3=random.random()*.01
        prev_mse=float('inf')
        alpha=.1
        prevs=(0,0,0,0)
        for i in range(1000000):
            cur_mse=mse(c1,c2,k1,k2,k3,data)
            if cur_mse>prev_mse:
                break
            prev_mse=cur_mse
            t,y=random.choice(data)
            prevs=(c1,c2,k1,k2)
            c1,c2,k1,k2,k3=gradient_step(c1,c2,k1,k2,k3,t,y,alpha)
            alpha*=.999
            #k3=0#Remove this after debugging
        if prev_mse<best_mse:
            c1,c2,k1,k2=prevs
            bestc1=c1
            bestc2=c2
            bestk1=k1
            bestk2=k2
            bestk3=k3
            best_mse=prev_mse
    return bestc1,bestc2,bestk1,bestk2,bestk3,best_mse
    
def run_default():    
    random.seed(11)
    import sys
    filename='data/PLAPEG_VE822_intracranial.csv'
    if len(sys.argv)>1:
        filename=sys.argv[1]
    data=np.genfromtxt(filename,delimiter=',')[1:,:2]
    x=map(float,list(data[:,0]))
    y=map(float,list(np.average(data[:,1:],axis=1)))
    c1,c2,k1,k2,k3,sse=sgd(zip(x,y))
    show_data(c1,c2,k1,k2,k3,data)
    show_data(.855,.1448,.11898,76.84,0.0,data)
    return (c1,c2,k1,k2,k3,sse)


if __name__=='__main__':
    print run_default()
