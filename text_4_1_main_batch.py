from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_AG_GP_utils import*
from text_4_1_dataset import*
from text_4_1_plot import*
from text_4_1_plot_logx import*
from text_4_1_GP_optimizer import*
from text_4_1_run import*

def AG_main_batch(train_data,test_data,init_point,index,iter=500,x_gt=[1,1],lr=[1e-3,1e-3],lambda_w=1,filename=None):

    D_x=len(x_gt)
    D=len(train_data)
    x0=init_point[0:D_x]
    w0=project_simplex(init_point[D_x:D_x+D])

    def loss_AG(x,index):
        x_=x[0:D_x]
        w_=x[D_x:D_x+D]
        loss=0
        for i in range(0,D):
            loss=loss+w_[i]*loss_for_D_index(x_,train_data[i],index[i])
        loss=loss-lambda_w*np.linalg.norm(w_-1.0/D*np.ones(D))**2
        return loss

    print("##################################################################")
    print("AG method")
    time_start=time.time()
    x_opt,AG_iter_res,AG_time=AG_run_batch(loss_AG,x0,w0,index,step=[0.001,0.001],lr=lr,iter=iter,inner_iter=1)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    if filename==None:
        np.savez("AG_4_1.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    else:
        np.savez("AG_4_1_"+filename+".npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return train_data,test_data

def Average_main_batch(train_data,test_data,init_point,index,iter=500,x_gt=[1,1],lr=1e-3,filename=None):

    D_x=len(x_gt)
    D=len(train_data)
    #x0=np.zeros(D_x)
    #w0=1.0/D*np.ones(D)
    x0=init_point[0:D_x]
    w0=project_simplex(init_point[D_x:D_x+3])

    def loss_Average(x,index):
        loss=0
        for i in range(0,D):
            loss=loss+w0[i]*loss_for_D_index(x,train_data[i],index[i])
        return loss

    print("##################################################################")
    print("Average method")
    time_start=time.time()
    x_opt,Average_iter_res,Average_time=Average_run_batch(loss_Average,x0,index,step=0.001,lr=lr,iter=iter)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of Average:',time_end-time_start,"s")

    if filename==None:
        np.savez("Average_4_1.npz",x_gt=x_gt,Average_iter_res=Average_iter_res,Average_time=Average_time)
    else:
        np.savez("Average_4_1_"+filename+".npz",x_gt=x_gt,Average_iter_res=Average_iter_res,Average_time=Average_time)
    return train_data,test_data

def FO_main_batch(train_data,test_data,init_point,index,iter=500,x_gt=[1,1],lr=[1e-3,1e-3],lambda_w=1,filename=None):
    
    D_x=len(x_gt)
    D=len(train_data)
    #x0=np.zeros(D_x)
    #w0=1.0/D*np.ones(D)
    x0=init_point[0:D_x]
    w0=project_simplex(init_point[D_x:D_x+D])

    def loss_FO(x,w,index):
        loss=0
        for i in range(0,D):
            loss=loss+w[i]*loss_for_D_index(x,train_data[i],index[i])
        loss=loss-lambda_w*np.linalg.norm(w-1.0/D*np.ones(D))**2
        return loss

    print("##################################################################")
    print("First order method")
    time_start=time.time()
    x_opt,FO_iter_res,FO_time=FO_run_batch(loss_FO,train_data,x0,w0,index,lambda_w,lr=lr,iter=iter)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of first order:',time_end-time_start,"s")

    if filename==None:
        np.savez("FO_4_1.npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    else:
        np.savez("FO_4_1_"+filename+".npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    return train_data,test_data

def main_one_time_batch(D_x=100,x_gt0=1,b=100,init_x0=0,iter=200,alpha=1e-3,beta=1e-3,lambda_w=1e-3,regenerate=False,heter_mean=True):
    x_gt=x_gt0*np.ones(D_x)
    init_point=init_x0*np.ones(D_x+3)

    if regenerate:
        generate_index(length=700,b=b,iter=iter,num_of_dataset=3)
        if heter_mean:
            generate_all_dataset_heter_mean(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)
        else:
            generate_all_dataset(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    AG_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w)
    Average_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,lr=alpha,x_gt=x_gt)
    FO_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w)

    AG_Average_FO_train_test_time_sc_plot_all(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

def main_multitimes_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_w=1e-3,regenerate=False,heter_mean=True):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter,num_of_dataset=3)
        if heter_mean:
            generate_all_dataset_heter_mean(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)
        else:
            generate_all_dataset(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,times):
        init_point=np.random.normal(0, 1, D_x+3)
        init_point[D_x]=1/3
        init_point[D_x+1]=1/3
        init_point[D_x+2]=1/3
        AG_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w,filename=str(i))
        Average_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,lr=alpha,x_gt=x_gt,filename=str(i))
        FO_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w,filename=str(i))

    multiplot_all(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta,times=times)

def main_multitimes_logx_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_w=1e-3,regenerate=False,heter_mean=True):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter,num_of_dataset=3)
        if heter_mean:
            generate_all_dataset_heter_mean(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)
        else:
            generate_all_dataset(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,times):
        init_point=np.random.normal(0, 1, D_x+3)
        init_point[D_x]=1/3
        init_point[D_x+1]=1/3
        init_point[D_x+2]=1/3
        AG_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w,filename=str(i))
        Average_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,lr=alpha,x_gt=x_gt,filename=str(i))
        FO_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w,filename=str(i))

    multiplot_all_logx(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta,times=times)

def main_multilambda_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_w=[1e-3,1e-1,1e+1],regenerate=False,heter_mean=True):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter,num_of_dataset=3)
        if heter_mean:
            generate_all_dataset_heter_mean(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)
        else:
            generate_all_dataset(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,len(lambda_w)):
        for j in range(0,times):
            init_point=np.random.normal(0, 1, D_x+3)
            init_point[D_x]=1/3
            init_point[D_x+1]=1/3
            init_point[D_x+2]=1/3
            AG_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w[i],filename="lambda_"+str(lambda_w[i])+"_"+str(j))
    multilambda_plot_all(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta,times=times)

def main_multilambda_logx_batch(D_x=100,x_gt0=1,times=10,b=100,iter=200,alpha=1e-3,beta=1e-3,lambda_w=[1e-3,1e-1,1e+1],regenerate=False,heter_mean=True):
    x_gt=x_gt0*np.ones(D_x)

    if regenerate:
        generate_index(length=700,b=b,iter=iter,num_of_dataset=3)
        if heter_mean:
            generate_all_dataset_heter_mean(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)
        else:
            generate_all_dataset(x_gt)#run when x_gt is changed
            save_train_and_test_data()#run when x_gt is changed
            time.sleep(5)

    train_data,test_data=load_train_and_test_data()
    index=load_index()

    for i in range(0,len(lambda_w)):
        for j in range(0,times):
            init_point=np.random.normal(0, 1, D_x+3)
            init_point[D_x]=1/3
            init_point[D_x+1]=1/3
            init_point[D_x+2]=1/3
            AG_main_batch(train_data,test_data,init_point=init_point,index=index,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w[i],filename="lambda_"+str(lambda_w[i])+"_"+str(j))
    multilambda_plot_all_logx(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta,times=times)

if __name__=="__main__":
    #main_one_time_batch(D_x=100,x_gt0=1,b=100,init_x0=0,iter=500,alpha=3e-1,beta=3e-3,lambda_w=1e-4,regenerate=True,heter_mean=True)

    #main_multitimes_batch(D_x=100,x_gt0=1,times=10,b=100,iter=500,alpha=3e-1,beta=3e-3,lambda_w=1e-2,regenerate=False,heter_mean=True)
    #main_multilambda_batch(D_x=100,x_gt0=1,times=10,b=100,iter=500,alpha=3e-1,beta=3e-3,lambda_w=[1e-4,1e-1,1e+2],regenerate=False,heter_mean=True)

    main_multitimes_logx_batch(D_x=100,x_gt0=1,times=10,b=100,iter=500,alpha=3e-1,beta=1e-1,lambda_w=1e-5,regenerate=False,heter_mean=True)
    main_multilambda_logx_batch(D_x=100,x_gt0=1,times=10,b=100,iter=500,alpha=3e-1,beta=1e-1,lambda_w=[1e-7,1e-5,1e-3],regenerate=False,heter_mean=True)
