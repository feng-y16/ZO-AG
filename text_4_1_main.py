from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_AG_GP_utils import*
from text_4_1_dataset import*
from text_4_1_plot import train_test_time_sc_plot,AG_Average_train_test_time_sc_plot,AG_train_test_time_sc_plot

def AG_main(train_data,test_data,x_gt=[1,1],lambda_w=1):

    D_x=len(x_gt)
    D=len(train_data)
    x0=np.zeros(D_x)
    w0=1.0/D*np.ones(D)

    def loss_AG(x):
        x_=x[0:D_x]
        w_=x[D_x:D_x+D]
        loss=0
        for i in range(0,D):
            loss=loss+w_[i]*loss_for_D(x_,train_data[i])
        loss=loss+lambda_w*np.linalg.norm(w_-1.0/D*np.ones(D))#加入正则化项
        return loss

    print("##################################################################")
    print("AG method")
    time_start=time.time()
    x_opt,AG_iter_res,AG_time=AG_run(loss_AG,x0,w0,step=[0.01,0.01],lr=[1e-4,1e-4],iter=500,inner_iter=1)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    np.savez("AG_4_1.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return train_data,test_data

def Average_main(train_data,test_data,x_gt=[1,1]):

    D_x=len(x_gt)
    D=len(train_data)
    x0=np.zeros(D_x)
    w0=1.0/D*np.ones(D)

    def loss_AG(x):
        x_=x[0:D_x]
        loss=0
        for i in range(0,D):
            loss=loss+w0[i]*loss_for_D(x_,train_data[i])
        return loss

    print("##################################################################")
    print("Average method")
    time_start=time.time()
    x_opt,Average_iter_res,Average_time=Average_run(loss_AG,x0,step=0.01,lr=1e-4,iter=500)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of Average:',time_end-time_start,"s")

    np.savez("Average_4_1.npz",x_gt=x_gt,Average_iter_res=Average_iter_res,Average_time=Average_time)
    return train_data,test_data

def FO_main(train_data,test_data,x_gt=[1,1],lambda_w=1):
    
    D_x=len(x_gt)
    D=len(train_data)
    x0=np.zeros(D_x)
    w0=1.0/D*np.ones(D)

    def loss_FO(x,w):
        loss=0
        for i in range(0,D):
            loss=loss+w[i]*loss_for_D(x,train_data[i])
        loss=loss+lambda_w*np.linalg.norm(w-1.0/D*np.ones(D))#加入正则化项
        return loss

    print("##################################################################")
    print("First order method")
    time_start=time.time()
    x_opt,FO_iter_res,FO_time=FO_run(loss_FO,train_data,x0,w0,lambda_w,lr=[1e-2,1e-2],iter=500)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of first order:',time_end-time_start,"s")

    np.savez("FO_4_1.npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    return train_data,test_data

if __name__=="__main__":
    x_gt=np.ones(100)
    lambda_w=1
    beta=1

    #generate_all_dataset(x_gt)#更改x_gt才需要运行
    #save_train_and_test_data()#更改x_gt才需要运行

    train_data,test_data=load_train_and_test_data()

    AG_main(train_data,test_data,x_gt=x_gt,lambda_w=lambda_w)
    Average_main(train_data,test_data,x_gt=x_gt)
    FO_main(train_data,test_data,x_gt=x_gt,lambda_w=lambda_w)

    #AG_train_test_time_sc_plot(train_data,test_data,lambda_w=lambda_w,beta=beta)
    #AG_Average_train_test_time_sc_plot(train_data,test_data,lambda_w=lambda_w,beta=beta)
    train_test_time_sc_plot(train_data,test_data,lambda_w=lambda_w,beta=beta)