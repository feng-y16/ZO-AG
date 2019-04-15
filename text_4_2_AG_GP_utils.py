from __future__ import print_function
import numpy as np
import random
import time
from text_4_2_dataset import sigmoid_truncated

def loss_function(delta,x,lambda_x,data):#compute loss for a dataset
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    a=data[:,0:length-1]
    c=data[:,length-1]
    #print(delta)
    h=1.0/(1+np.exp(-a.dot(x+delta)))
    value=(c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)))/num-lambda_x*np.linalg.norm(x,2)**2
    return value

def loss_function_index(delta,x,lambda_x,data,index):#compute loss for a dataset
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    #print(index)
    index=list(map(int,index))
    a=data[index,0:length-1]
    c=data[index,length-1]
    #print(delta)
    h=1.0/(1+np.exp(-a.dot(x+delta)))
    value=(c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)))/len(index)-lambda_x*np.linalg.norm(x,2)**2
    return value

def acc_for_D(delta,x,lambda_x,data):#compute loss for a dataset
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    acc=0
    for i in range(0,np.shape(data)[0]):
        if abs(c[i]-sigmoid_truncated((a[i].T).dot(x+delta)))<1e-2:
            acc=acc+1
    acc=acc/np.shape(data)[0]
    return acc

def loss_derivative_delta(delta,x,lambda_x,data):
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=1.0/(1+np.exp(-a.dot(x+delta)))
    derivative=-(((h-c).T).dot(a)).T/num
    #print(derivative)
    return derivative

def loss_derivative_delta_index(delta,x,lambda_x,data,index):
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    index=list(map(int,index))
    a=data[index,0:length-1]
    c=data[index,length-1]
    h=1.0/(1+np.exp(-a.dot(x+delta)))
    derivative=-(((h-c).T).dot(a)).T/len(index)
    #print(derivative)
    return derivative

def loss_derivative_x(delta,x,lambda_x,data):
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=1.0/(1+np.exp(-a.dot(x+delta)))
    derivative=-(((h-c).T).dot(a)).T/num-2*lambda_x*x
    #print(derivative)
    return derivative

def loss_derivative_x_index(delta,x,lambda_x,data,index):
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    index=list(map(int,index))
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=1.0/(1+np.exp(-a.dot(x+delta)))
    derivative=-(((h-c).T).dot(a)).T/len(index)-2*lambda_x*x
    #print(derivative)
    return derivative

def project_simplex(x):
    lambda_opt=np.zeros(len(x))
    x_temp=x
    while(True):
        x_temp=x_temp-lambda_opt
        flag=0
        sum=0
        for i in range(0,len(x)):
            if x_temp[i]>=0:
                flag=flag+1
                sum=sum+x_temp[i]
        if abs(sum-1)<1e-6:
            break
        elif sum==0:
            lambda_opt=(np.sum(x)-1)/len(x)
        else:
            lambda_opt=(sum-1)/flag*np.ones(len(x))
    x_temp[x_temp<0]=0
    return x_temp

def project_bound(x,bound):
    D=len(x)
    flag=0
    for i in range(0,D):
        if x[i]<bound[i][0]:
            x[i]=bound[i][0]
            flag=1
            continue
        if x[i]>bound[i][1]:
            x[i]=bound[i][1]
            flag=1
            continue
    return x,flag

def project_inf(x,epsilon):
    D=len(x)
    flag=0
    distance=np.linalg.norm(x, ord=np.inf)
    if distance>epsilon:
        flag=1
        x=x/distance*epsilon
    return x,flag

def project_f_l2(x,x_cen,epsilon):
    D=len(x)
    flag=0
    distance=np.linalg.norm(x-x_cen, ord=2)
    if distance>epsilon:
        flag=1
        x=x_cen+epsilon*(x-x_cen)/distance
    return x,flag

def ZOSIGNSGD_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10,project=project_bound):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q

        x_temp,flag2=project(x_opt - lr * np.sign(dx), bound)

        y_temp=func(x_temp)

        if np.isnan(y_temp) == 1:
            test = 1

        if i%10 == 0:
            print("ZOsignSGD for -likelihood: Iter = %d, obj = %3.4f" % (i, y_temp) )
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt

def ZOPSGD(func,x0,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=x_opt - lr *  dx
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
            #if flag%2==0:
            #    lr=lr*0.98
    return x_opt

def ZOPSGA(func,x0,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=x_opt+lr*dx
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
            #if flag%2==0:
            #    lr=lr*0.98
    return x_opt

def ZOPSGD_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10,project=project_bound):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project(x_opt-lr*dx,bound)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt

def ZOPSGA_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10,project=project_bound):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project(x_opt+lr*dx,bound)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt

def ZOPSGD_bounded_f(func,x0,dis_f,epsilon,step,x_cen,lr=0.1,iter=100,Q=10,project=project_f_l2):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project(x_opt-lr*(dx),x_cen,epsilon)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt

def ZOPSGA_bounded_f(func,x0,dis_f,epsilon,step,x_cen,lr=0.1,iter=100,Q=10,project=project_f_l2):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size

    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project(x_opt+lr*dx,x_cen,epsilon)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt

def ZOPSGA_simplex(func,x0,step,lr=0.1,iter=100,Q=10,project=project_simplex):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=project(x_opt+lr*dx)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt