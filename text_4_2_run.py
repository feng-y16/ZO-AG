from __future__ import print_function
import numpy as np
import random
import time
from text_4_2_dataset import sigmoid_truncated
from text_4_2_AG_GP_utils import*

def AG_minmax(func,delta0,x0,step,lr,epsilon,iter=20,inner_iter=1):
    delta_opt=delta0
    x_opt=x0
    D_d=len(delta0)
    D_x=len(x0)
    flag=0
    best_f=func(delta0,x0)
    AG_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        AG_iter_res[i][0:len(delta0)] = delta_opt
        AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        def func_deltafixed(x):
            return func(delta_opt,x)
        x_opt=ZOPSGA(func_deltafixed,x_opt,step[1],lr[1],inner_iter)

        def func_xfixed(delta):
            return func(delta,x_opt)
        delta_opt=ZOPSGD_bounded(func_xfixed,delta_opt,epsilon,step[0],lr[0],inner_iter)

        #print(x_opt)
        #print(delta_opt)
        temp_f=func_xfixed(delta_opt)
        if i%10 == 0:
            print("ZO-AG for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], temp_f) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
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

def AG_run(func,delta0,x0,epsilon,step,lr,iter=20,inner_iter=1):
    D_d=len(delta0)
    bound_delta=epsilon*np.ones((D_d,2))
    bound_delta[:,0]=-bound_delta[:,1]
    x_opt,AG_iter_res,AG_time=AG_minmax(func,delta0,x0,step,lr,bound_delta,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def FO_run(func,data,delta0,x0,epsilon,lambda_x,lr,iter=100,project=project_inf):
    lr=np.array(lr)
    FO_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    FO_time=np.zeros(iter)
    D_d=len(delta0)
    D_x=len(x0)
    delta_opt=delta0
    x_opt=x0
    best_f=func(delta0,x0)
    sigma=1
    flag1=0

    for i in range(0,iter):
        FO_time[i]=time.time()
        FO_iter_res[i][0:D_d]=delta_opt
        FO_iter_res[i][D_d:D_d+D_x]=x_opt

        dx=loss_derivative_x(delta_opt,x_opt,lambda_x,data)
        x_temp=x_opt+dx*lr[1]
        y_temp=func(delta_opt,x_temp)
        if y_temp>func(delta_opt,x_opt):
            x_opt=x_temp

        ddelta=np.zeros(D_x)
        ddelta=loss_derivative_delta(delta_opt,x_opt,lambda_x,data)
        delta_temp,flag0=project(delta_opt-ddelta*lr[0],epsilon)
        y_temp=func(delta_temp,x_opt)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if i%10 == 0:
            print("FO for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
        if y_temp<func(delta_opt,x_opt):
            best_f=y_temp
            delta_opt=delta_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt,FO_iter_res,FO_time

def BL_run(func,data,x0,lambda_x,lr,iter=100):
    lr=np.array(lr)
    BL_iter_res=np.zeros((iter,2*len(x0)))
    BL_time=np.zeros(iter)
    D_x=len(x0)
    D_d=D_x
    delta_opt=np.zeros(D_d)
    x_opt=x0
    best_f=func(x_opt)
    sigma=1
    flag1=0

    for i in range(0,iter):
        BL_time[i]=time.time()
        BL_iter_res[i][0:D_d]=delta_opt
        BL_iter_res[i][D_d:D_d+D_x]=x_opt

        dx=loss_derivative_x(delta_opt,x_opt,lambda_x,data)
        x_temp=x_opt+dx*lr[1]
        y_temp=func(x_temp)
        if y_temp>func(x_opt):
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if i%10 == 0:
            print("BL for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
    return x_opt,BL_iter_res,BL_time

def AG_minmax_batch(func,delta0,x0,index,step,lr,epsilon,iter=20,inner_iter=1):
    delta_opt=delta0
    x_opt=x0
    D_d=len(delta0)
    D_x=len(x0)
    flag=0
    best_f=func(delta0,x0,index[0])
    AG_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        AG_iter_res[i][0:len(delta0)] = delta_opt
        AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        def func_deltafixed(x):
            return func(delta_opt,x,index[i])
        x_opt=ZOPSGA(func_deltafixed,x_opt,step[1],lr[1],inner_iter)

        def func_xfixed(delta):
            return func(delta,x_opt,index[i])
        delta_opt=ZOPSGD_bounded(func_xfixed,delta_opt,epsilon,step[0],lr[0],inner_iter)

        temp_f=func_xfixed(delta_opt)
        if i%10 == 0:
            print("ZO-AG for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], temp_f) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
        if temp_f<best_f:
            best_f=temp_f
        else:
            flag=flag+1
            #if flag%3==0:
            #    lr[0]=lr[0]*0.98
    return x_opt,AG_iter_res,AG_time

def AG_run_batch(func,delta0,x0,index,epsilon,step,lr,iter=20,inner_iter=1):
    D_d=len(delta0)
    bound_delta=epsilon*np.ones((D_d,2))
    bound_delta[:,0]=-bound_delta[:,1]
    x_opt,AG_iter_res,AG_time=AG_minmax_batch(func,delta0,x0,index,step,lr,bound_delta,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def FO_run_batch(func,data,delta0,x0,index,epsilon,lambda_x,lr,iter=100,project=project_inf):
    lr=np.array(lr)
    FO_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    FO_time=np.zeros(iter)
    D_d=len(delta0)
    D_x=len(x0)
    delta_opt=delta0
    x_opt=x0
    best_f=func(delta0,x0,index[0])
    sigma=1
    flag1=0

    for i in range(0,iter):
        FO_time[i]=time.time()
        FO_iter_res[i][0:D_d]=delta_opt
        FO_iter_res[i][D_d:D_d+D_x]=x_opt

        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,data,index[i])
        x_temp=x_opt+dx*lr[1]
        y_temp=func(delta_opt,x_temp,index[i])
        if y_temp>func(delta_opt,x_opt,index[i]):
            x_opt=x_temp

        ddelta=np.zeros(D_x)
        ddelta=loss_derivative_delta_index(delta_opt,x_opt,lambda_x,data,index[i])
        delta_temp,flag0=project(delta_opt-ddelta*lr[0],epsilon)
        y_temp=func(delta_temp,x_opt,index[i])
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if i%10 == 0:
            print("FO for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
        if y_temp<func(delta_opt,x_opt,index[i]):
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt,FO_iter_res,FO_time

def BL_run_batch(func,data,x0,index,lambda_x,lr,iter=100):
    lr=np.array(lr)
    BL_iter_res=np.zeros((iter,2*len(x0)))
    BL_time=np.zeros(iter)
    D_x=len(x0)
    D_d=D_x
    delta_opt=np.zeros(D_d)
    x_opt=x0
    best_f=func(x_opt,index[0])
    sigma=1
    flag1=0

    for i in range(0,iter):
        BL_time[i]=time.time()
        BL_iter_res[i][0:D_d]=delta_opt
        BL_iter_res[i][D_d:D_d+D_x]=x_opt

        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,data,index[i])
        x_temp=x_opt+dx*lr[1]
        y_temp=func(x_temp,index[i])
        if y_temp>func(x_opt,index[i]):
            x_opt=x_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if i%10 == 0:
            print("BL for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
    return x_opt,BL_iter_res,BL_time