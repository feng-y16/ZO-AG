from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_dataset import sigmoid_truncated
from text_4_1_AG_GP_utils import*


def AG_minmax_bounded_simplex(func,x0,y0,step,lr,bound_x,iter=20,inner_iter=1):
    x_opt=x0
    y_opt=y0
    D_x=len(x0)
    D_y=len(y0)
    flag=0
    best_f=func(np.hstack((x0,y0)))
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
            #if flag%3==0:
            #    lr[0]=lr[0]*0.98
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
            #if flag1%3==0:
            #    lr=lr*0.98
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
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt,FO_iter_res,FO_time



def AG_minmax_bounded_simplex_batch(func,x0,y0,index,step,lr,bound_x,iter=20,inner_iter=1):
    x_opt=x0
    y_opt=y0
    D_x=len(x0)
    D_y=len(y0)
    flag=0
    best_f=func(np.hstack((x0,y0)),index[0])
    AG_iter_res=np.zeros((iter,len(x0)+len(y0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        AG_iter_res[i][0:len(x0)] = x_opt
        AG_iter_res[i][len(x0):len(x0)+len(y0)] = y_opt

        def func_xfixed(y):
            return func(np.hstack((x_opt,y)),index[i])
        y_opt=ZOPSGA_simplex(func_xfixed,y_opt,step[1],lr[1],inner_iter)

        def func_yfixed(x):
            return func(np.hstack((x,y_opt)),index[i])
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
            #if flag%3==0:
            #    lr[0]=lr[0]*0.98
    return x_opt,AG_iter_res,AG_time

def AG_run_batch(func,x0,y0,index,step,lr,iter=20,inner_iter=1):
    D_x=len(x0)
    bound_x=1000000*np.ones((D_x,2))
    bound_x[:,0]=-bound_x[:,1]
    x_opt,AG_iter_res,AG_time=AG_minmax_bounded_simplex_batch(func,x0,y0,index,step,lr,bound_x,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def Average_run_batch(func,x0,index,step,D_w=3,lr=0.1,iter=100,Q=10,project=project_bound):
    D_x=len(x0)
    bound=1000000*np.ones((D_x,2))
    bound[:,0]=-bound[:,1]

    Average_iter_res=np.zeros((iter,len(x0)+D_w))
    Average_time=np.zeros(iter)
    
    D=len(x0)
    x_opt=x0
    best_f=func(x0,index[0])
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
            grad=D*(func(x_opt+u*step,index[i])-func(x_opt,index[i]))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project(x_opt-lr*dx,bound)
        y_temp=func(x_temp,index[i])
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
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt,Average_iter_res,Average_time

def FO_run_batch(func,data_all,x0,w0,index,lambda_w,lr,iter=100,project=project_simplex):
    lr=np.array(lr)
    FO_iter_res=np.zeros((iter,len(x0)+len(w0)))
    FO_time=np.zeros(iter)
    D_x=len(x0)
    D_w=len(w0)
    x_opt=x0
    w_opt=w0
    best_f=func(x0,w0,index[0])
    sigma=1
    flag1=0

    for i in range(0,iter):
        FO_time[i]=time.time()
        FO_iter_res[i][0:D_x]=x_opt
        FO_iter_res[i][D_x:D_x+D_w]=w_opt

        dw=loss_derivative_w_index(x_opt,w_opt,lambda_w,data_all,index[i])
        w_temp=project(w_opt+dw*lr[1])
        y_temp=func(x_opt,w_temp,index[i])
        if y_temp>func(x_opt,w_opt,index[i]):
            w_opt=w_temp

        dx=np.zeros(D_x)
        for j in range(0,D_w):
            dx=dx+w_opt[j]*loss_derivative_x_for_D_index(x_opt,data_all[j],index[i][j])
        x_temp=x_opt-dx*lr[0]
        y_temp=func(x_temp,w_opt,index[i])
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
        if y_temp<func(x_opt,w_opt,index[i]):
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt,FO_iter_res,FO_time