from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_dataset import sigmoid_truncated

def loss_for_D(x,data):#compute loss for a dataset
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=1.0/(1+np.exp(-a.dot(x)))
    loss=-(c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)))
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

def loss_derivative_x_for_D(x,data):
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=1.0/(1+np.exp(-a.dot(x)))
    derivative=(((h-c).T).dot(a)).T
    #print(derivative)
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
            if flag1%3==0:
                lr=lr*0.98
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
            if flag%2==0:
                lr=lr*0.98
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
            if flag%2==0:
                lr=lr*0.98
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
            if flag1%3==0:
                lr=lr*0.98
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
            if flag1%3==0:
                lr=lr*0.98
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
            if flag1%3==0:
                lr=lr*0.98
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
            if flag1%3==0:
                lr=lr*0.98
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
            if flag1%3==0:
                lr=lr*0.98
    return x_opt

def AG_minmax_bounded_simplex(func,x0,y0,step,lr,bound_x,iter=20,inner_iter=1):
    x_opt=x0
    y_opt=y0
    D_x=len(x0)
    D_y=len(y0)
    flag=0
    best_f=1000000
    AG_iter_res=np.zeros((iter,len(x0)+len(y0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        AG_iter_res[i][0:len(x0)] = x_opt
        AG_iter_res[i][len(x0):len(x0)+len(y0)] = y_opt

        def func_xfixed(y):
            return func(np.hstack((x_opt,y)))
        y_opt=ZOPSGA_simplex(func_xfixed,y_opt,step[1],lr[1],inner_iter)

        def func_yfixed(x):
            return func(np.hstack((x,y_opt)))
        x_opt=ZOPSGD_bounded(func_yfixed,x_opt,bound_x,step[0],lr[0],inner_iter)

        temp_f=func_yfixed(x_opt)
        if i%10 == 0:
            print("ZO-AG for Min-Max: Iter = %d, lr_x=%f, obj = %3.4f" % (i, lr[0], temp_f) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
        #print("x_opt=",end="")
        #print(x_opt)
        #print("step_x=",end="")
        #print(step[0])
        #print("lr_x=",end="")
        #print(lr[0])
        if temp_f<best_f:
            best_f=temp_f
        else:
            flag=flag+1
            if flag%3==0:
                lr[0]=lr[0]*0.98
    return x_opt,AG_iter_res,AG_time

def AG_run(func,x0,y0,step,lr,iter=20,inner_iter=1):
    D_x=len(x0)
    bound_x=1000000*np.ones((D_x,2))
    bound_x[:,0]=-bound_x[:,1]
    x_opt,AG_iter_res,AG_time=AG_minmax_bounded_simplex(func,x0,y0,step,lr,bound_x,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def Average_run(func,x0,step,D_w=3,lr=0.1,iter=100,Q=10,project=project_bound):
    D_x=len(x0)
    bound=1000000*np.ones((D_x,2))
    bound[:,0]=-bound[:,1]

    Average_iter_res=np.zeros((iter,len(x0)+D_w))
    Average_time=np.zeros(iter)
    
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        Average_time[i]=time.time()
        Average_iter_res[i][0:D]=x_opt
        Average_iter_res[i][D:D+D_w]=1.0/D_w*np.ones(D_w)
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
        if i%10 == 0:
            print("Average for Min-Max: Iter = %d, lr_x=%f, obj = %3.4f" % (i, lr, y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            if flag1%3==0:
                lr=lr*0.98
    return x_opt,Average_iter_res,Average_time

def FO_run(func,data_all,x0,w0,lambda_w,lr,iter=100,project=project_simplex):
    lr=np.array(lr)
    FO_iter_res=np.zeros((iter,len(x0)+len(w0)))
    FO_time=np.zeros(iter)
    D_x=len(x0)
    D_w=len(w0)
    x_opt=x0
    w_opt=w0
    best_f=func(x0,w0)
    sigma=1
    flag1=0

    for i in range(0,iter):
        FO_time[i]=time.time()
        FO_iter_res[i][0:D_x]=x_opt
        FO_iter_res[i][D_x:D_x+D_w]=w_opt

        dw=loss_derivative_w(x_opt,w_opt,lambda_w,data_all)
        w_temp=project(w_opt+dw*lr[1])
        y_temp=func(x_opt,w_temp)
        if y_temp>func(x_opt,w_opt):
            w_opt=w_temp

        dx=np.zeros(D_x)
        for j in range(0,D_w):
            dx=dx+w_opt[j]*loss_derivative_x_for_D(x_opt,data_all[j])
        x_temp=x_opt-dx*lr[0]
        y_temp=func(x_temp,w_opt)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if i%10 == 0:
            print("FO for Min-Max: Iter = %d, lr_x=%f, obj = %3.4f" % (i, lr[0], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
        if y_temp<func(x_opt,w_opt):
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            if flag1%3==0:
                lr=lr*0.98
    return x_opt,FO_iter_res,FO_time

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