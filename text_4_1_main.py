from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_AG_GP_utils import*
from text_4_1_dataset import*
from text_4_1_plot import train_test_time_sc_plot,AG_Average_train_test_time_sc_plot,AG_train_test_time_sc_plot,AG_Average_FO_train_test_time_sc_plot
from text_4_1_GP_optimizer import*

def AG_main(train_data,test_data,init_point,iter=500,x_gt=[1,1],lr=[1e-4,1e-4],lambda_w=1):

    D_x=len(x_gt)
    D=len(train_data)
    #x0=np.zeros(D_x)
    #w0=1.0/D*np.ones(D)
    x0=init_point[0:D_x]
    w0=project_simplex(init_point[D_x:D_x+D])

    def loss_AG(x):
        x_=x[0:D_x]
        w_=x[D_x:D_x+D]
        loss=0
        for i in range(0,D):
            loss=loss+w_[i]*loss_for_D(x_,train_data[i])
        loss=loss+lambda_w*np.linalg.norm(w_-1.0/D*np.ones(D))
        return loss

    print("##################################################################")
    print("AG method")
    time_start=time.time()
    x_opt,AG_iter_res,AG_time=AG_run(loss_AG,x0,w0,step=[0.01,0.01],lr=lr,iter=iter,inner_iter=1)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    np.savez("AG_4_1.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return train_data,test_data

def Average_main(train_data,test_data,init_point,iter=500,x_gt=[1,1],lr=1e-4):

    D_x=len(x_gt)
    D=len(train_data)
    #x0=np.zeros(D_x)
    #w0=1.0/D*np.ones(D)
    x0=init_point[0:D_x]
    w0=project_simplex(init_point[D_x:D_x+D])

    def loss_AG(x):
        x_=x[0:D_x]
        loss=0
        for i in range(0,D):
            loss=loss+w0[i]*loss_for_D(x_,train_data[i])
        return loss

    print("##################################################################")
    print("Average method")
    time_start=time.time()
    x_opt,Average_iter_res,Average_time=Average_run(loss_AG,x0,step=0.01,lr=lr,iter=iter)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of Average:',time_end-time_start,"s")

    np.savez("Average_4_1.npz",x_gt=x_gt,Average_iter_res=Average_iter_res,Average_time=Average_time)
    return train_data,test_data

def FO_main(train_data,test_data,init_point,iter=500,x_gt=[1,1],lr=[1e-4,1e-4],lambda_w=1):
    
    D_x=len(x_gt)
    D=len(train_data)
    #x0=np.zeros(D_x)
    #w0=1.0/D*np.ones(D)
    x0=init_point[0:D_x]
    w0=project_simplex(init_point[D_x:D_x+D])

    def loss_FO(x,w):
        loss=0
        for i in range(0,D):
            loss=loss+w[i]*loss_for_D(x,train_data[i])
        loss=loss+lambda_w*np.linalg.norm(w-1.0/D*np.ones(D))
        return loss

    print("##################################################################")
    print("First order method")
    time_start=time.time()
    x_opt,FO_iter_res,FO_time=FO_run(loss_FO,train_data,x0,w0,lambda_w,lr=lr,iter=iter)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of first order:',time_end-time_start,"s")

    np.savez("FO_4_1.npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    return train_data,test_data

def GP_main(train_data,test_data,init_num=10,iter=500,lr=[1e-4,1e-4],inner_iter=50):
    print("##################################################################")
    print("STABLEOPT method")
    time_start=time.time()
    optimizer=STABLEOPT(train_data=train_data,test_data=test_data,beta=4*np.ones(iter+init_num),init_num=init_num,mu0=0,iter=inner_iter,step=[0.01,0.01],lr=lr)
    optimizer.run()
    time_end=time.time()
    print('Time cost of STABLEOPT:',time_end-time_start,"s")

    np.savez("GP_4_1.npz",x_gt=x_gt,GP_iter_res=optimizer.X[optimizer.init_num:],GP_time=optimizer.GP_time)
    return train_data,test_data,optimizer.iter_initial_point

if __name__=="__main__":
    x_gt=np.ones(100)
    lambda_w=1
    alpha=1e-4
    beta=1e-4
    init_num=1
    iter=200
    init_point=np.zeros(103)#103=100+3, is used when do not run GP_main, but want to run other mains  

    generate_all_dataset(x_gt)#run when x_gt is changed
    save_train_and_test_data()#run when x_gt is changed

    train_data,test_data=load_train_and_test_data()

    #_,__,init_point=GP_main(train_data,test_data,init_num=init_num,iter=iter,lr=[alpha,beta],inner_iter=50)
    AG_main(train_data,test_data,init_point=init_point,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w)
    Average_main(train_data,test_data,init_point=init_point,iter=iter,lr=alpha,x_gt=x_gt)
    FO_main(train_data,test_data,init_point=init_point,iter=iter,x_gt=x_gt,lr=[alpha,beta],lambda_w=lambda_w)

    #AG_train_test_time_sc_plot(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)
    #AG_Average_train_test_time_sc_plot(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)
    AG_Average_FO_train_test_time_sc_plot(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    #train_test_time_sc_plot(train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)