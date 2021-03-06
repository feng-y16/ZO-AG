from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_dataset import sigmoid_truncated

def loss_for_D(x,data):#compute loss for a dataset
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=1.0/(1+np.exp(-a.dot(x)))
    loss=-(c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)))/num
    return loss

def loss_for_D_index(x,data,index):#compute loss for a dataset
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    index=list(map(int,index))
    a=data[index,0:length-1]
    c=data[index,length-1]
    h=1.0/(1+np.exp(-a.dot(x)))
    loss=-(c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)))/len(index)
    return loss

def acc_for_D(x,data):#compute loss for a dataset
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    acc=0
    for i in range(0,np.shape(data)[0]):
        if abs(c[i]-sigmoid_truncated((a[i].T).dot(x)))<1e-2:
            acc=acc+1
    acc=acc/np.shape(data)[0]
    return acc

def loss_derivative_x_for_D_index(x,data,index):
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    index=list(map(int,index))
    a=data[index,0:length-1]
    c=data[index,length-1]
    h=1.0/(1+np.exp(-a.dot(x)))
    derivative=(((h-c).T).dot(a)).T/len(index)
    #print(derivative)
    return derivative

def loss_derivative_x_for_D(x,data):
    length=np.shape(data)[1]
    num=np.shape(data)[0]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=1.0/(1+np.exp(-a.dot(x)))
    derivative=(((h-c).T).dot(a)).T/num
    #print(derivative)
    return derivative

def loss_derivative_w_index(x,w,lambda_w,data_all,index_all):
    derivative=np.zeros(len(w))
    for i in range(0,len(w)):
        #length=np.shape(data_all[i])[1]
        #a=data_all[i][:,0:length-1]
        #c=data_all[i][:,length-1]
        derivative[i]=loss_for_D_index(x,data_all[i],index_all[i])-2*lambda_w*(w[i]-1.0/len(w))
    return derivative

def loss_derivative_w(x,w,lambda_w,data_all):
    derivative=np.zeros(len(w))
    for i in range(0,len(w)):
        #length=np.shape(data_all[i])[1]
        #a=data_all[i][:,0:length-1]
        #c=data_all[i][:,length-1]
        derivative[i]=loss_for_D(x,data_all[i])-2*lambda_w*(w[i]-1.0/len(w))
    return derivative

def bisection(f,lambda_l,lambda_u,epsilon):
    while abs(lambda_u-lambda_l)>epsilon:
        lambda_m=(lambda_u+lambda_l)/2
        if f(lambda_m)==0:
            return lambda_m
        elif np.sign(f(lambda_m))==np.sign(f(lambda_l)):
            lambda_l=lambda_m
        else:
            lambda_u=lambda_m
    return (lambda_u+lambda_l)/2

def project_simplex_bisection(x):
    lambda_l=np.min(x)-1/len(x)
    lambda_u=np.max(x)-1/len(x)
    epsilon=1e-5
    def g(lambda_):
        g_value=0
        for i in range(0,len(x)):
            temp=x[i]-lambda_
            if temp>0:
                g_value=g_value+(x[i]-lambda_)
        return g_value-1
    x_temp=x-bisection(g,lambda_l,lambda_u,epsilon)
    x_temp[x_temp<0]=0
    return x_temp

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



if __name__=="__main__":
    m=20
    sigma=10
    num=10000
    a_test=[]
    for i in range(0,num):
        a_test.append(np.random.normal(0, sigma, m))
    
    result1=[]
    result2=[]
    time_start=time.time()
    for i in range(0,num):
        a=a_test[i]
        result1.append(project_simplex(a))
    time_end=time.time()
    print('Time cost of project_simplex:',time_end-time_start,"s")

    time_start=time.time()
    for i in range(0,num):
        a=a_test[i]
        result2.append(project_simplex_bisection(a))
    time_end=time.time()
    print('Time cost of project_simplex_bisection:',time_end-time_start,"s")

    distance=[]
    for i in range(0,num):
        distance.append(np.linalg.norm(result1[i]-result2[i]))
    print('Max distance:',np.max(np.array(distance)))