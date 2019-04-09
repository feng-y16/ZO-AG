from __future__ import print_function
import numpy as np
import random
import time
from text_4_2_AG_GP_utils import*
from text_4_2_dataset import*
from text_4_2_plot import*
from text_4_2_plot_logx import*
from text_4_2_run import*

def AG_main_batch(train_data,test_data,init_point,index,epsilon=0.5,iter=500,x_gt=[1,1],lr=[1e-3,1e-3],lambda_x=1,filename=None):

    D=len(x_gt)
    delta0,flag0=project_inf(init_point[0:D],epsilon)
    x0=init_point[D:2*D]

    def loss_AG(delta,x,index):
        loss=loss_function_index(delta,x,lambda_x,train_data,index)
        return loss

    print("##################################################################")
    print("AG method")
    time_start=time.time()
    x_opt,AG_iter_res,AG_time=AG_run_batch(loss_AG,delta0,x0,index,epsilon,step=[0.001,0.001],lr=lr,iter=iter,inner_iter=1)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    if filename==None:
        np.savez("AG_4_2.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    else:
        np.savez("AG_4_2_"+filename+".npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return train_data,test_data

def FO_main_batch(train_data,test_data,init_point,index,epsilon=0.5,iter=500,x_gt=[1,1],lr=[1e-3,1e-3],lambda_x=1,filename=None):
    
    D=len(x_gt)
    delta0,flag0=project_inf(init_point[0:D],epsilon)
    x0=init_point[D:2*D]

    def loss_FO(delta,x,index):
        loss=loss_function_index(delta,x,lambda_x,train_data,index)
        return loss

    print("##################################################################")
    print("First order method")
    time_start=time.time()
    x_opt,FO_iter_res,FO_time=FO_run_batch(loss_FO,train_data,delta0,x0,index,epsilon,lambda_x,lr=lr,iter=iter)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of first order:',time_end-time_start,"s")

    if filename==None:
        np.savez("FO_4_2.npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    else:
        np.savez("FO_4_2_"+filename+".npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    return train_data,test_data

def main_one_time_batch(D_x=100,x_gt0=1,b=100,init_x0=0,epsilon=0.5,iter=200,alpha=1e-3,beta=1e-3,lambda_x=1e-3,regenerate=False):
    x_gt=x_gt0*np.ones(D_x)
    init_point=init_x0*np.ones(2*D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter)
        generate_all_dataset(x_gt)
        save_train_and_test_data()
        time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    AG_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=epsilon,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x)
    FO_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=epsilon,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x)

    AG_FO_plot_all(train_data,test_data,lambda_x=lambda_x,alpha=alpha,beta=beta)

def main_multitimes_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_x=1e-3,regenerate=False):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter)
        generate_all_dataset(x_gt)
        save_train_and_test_data()
        time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,times):
        init_point=np.random.normal(0, 1, 2*D_x)
        AG_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=0.5,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x,filename=str(i))
        FO_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=0.5,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x,filename=str(i))

    multiplot_all(train_data,test_data,lambda_x=lambda_x,alpha=alpha,beta=beta,times=times)

def main_multitimes_logx_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_x=1e-3,regenerate=False):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter)
        generate_all_dataset(x_gt)
        save_train_and_test_data()
        time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,times):
        init_point=np.random.normal(0, 1, 2*D_x)
        AG_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=0.5,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x,filename=str(i))
        FO_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=0.5,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x,filename=str(i))

    multiplot_all_logx(train_data,test_data,lambda_x=lambda_x,alpha=alpha,beta=beta,times=times)

def main_multilambda_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_x=[1e-3,1e-1,1e+1],regenerate=False):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter)
        generate_all_dataset(x_gt)
        save_train_and_test_data()
        time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,len(lambda_x)):
        for j in range(0,times):
            init_point=np.random.normal(0, 1, 2*D_x)
            AG_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=0.5,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x[i],filename="lambda_"+str(lambda_x[i])+"_"+str(j))
    multilambda_plot_all(train_data,test_data,lambda_x=lambda_x,alpha=alpha,beta=beta,times=times)

def main_multilambda_logx_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_x=[1e-3,1e-1,1e+1],regenerate=False):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter)
        generate_all_dataset(x_gt)
        save_train_and_test_data()
        time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,len(lambda_x)):
        for j in range(0,times):
            init_point=np.random.normal(0, 1, 2*D_x)
            AG_main_batch(train_data,test_data,init_point=init_point,index=index,epsilon=0.5,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_x=lambda_x[i],filename="lambda_"+str(lambda_x[i])+"_"+str(j))
    multilambda_plot_all_logx(train_data,test_data,lambda_x=lambda_x,alpha=alpha,beta=beta,times=times)

if __name__=="__main__":
    #main_one_time_batch(D_x=100,x_gt0=1,b=100,init_x0=0,iter=50,alpha=3e-1,beta=3e-3,lambda_x=1e-4,regenerate=False)

    #main_multitimes_batch(D_x=100,x_gt0=1,times=2,b=100,iter=50,alpha=3e-1,beta=3e-3,lambda_x=1e-2,regenerate=False)
    #main_multilambda_batch(D_x=100,x_gt0=1,times=2,b=100,iter=50,alpha=3e-1,beta=3e-3,lambda_x=[1e-4,1e-1,1e+2],regenerate=False)

    #main_multitimes_logx_batch(D_x=100,x_gt0=1,times=2,b=100,iter=50,alpha=3e-1,beta=1e-1,lambda_x=1e-5,regenerate=False)
    main_multilambda_logx_batch(D_x=100,x_gt0=1,times=2,b=100,iter=50,alpha=3e-1,beta=1e-1,lambda_x=[1e-7,1e-5,1e-3],regenerate=False)

