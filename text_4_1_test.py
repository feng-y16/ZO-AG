from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_AG_GP_utils import*
from text_4_1_dataset import*
from text_4_1_plot import*
from text_4_1_GP_optimizer import*

def Average_test_plot(train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    Average=np.load("Average_4_1.npz")
    x_gt=Average['x_gt']
    Average_iter_res=Average['Average_iter_res']
    Average_time=Average['Average_time']
    D=len(train_data)
    iter=len(Average_time)

    train_loss=np.zeros(iter)
    train_accuracy=np.zeros(iter)
    test_accuracy=np.zeros(iter)
    for i in range(0,iter):
        train_loss[i]=loss_for_D(Average_iter_res[i][0:100],train_data)
        train_accuracy[i]=acc_for_D(Average_iter_res[i][0:100],train_data)
        test_accuracy[i]=acc_for_D(Average_iter_res[i][0:100],test_data)

    p1,=plt.plot(range(0,iter),train_loss)
    plt.xlabel("Number of iterations")
    plt.ylabel("Train loss")
    plt.show()

    p2,=plt.plot(range(0,iter),train_accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Train accuracy")
    plt.show()

    p3,=plt.plot(range(0,iter),test_accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Test accuracy")
    plt.show()

def FO_test_plot(train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    FO=np.load("FO_4_1.npz")
    x_gt=FO['x_gt']
    FO_iter_res=FO['FO_iter_res']
    FO_time=FO['FO_time']
    D=len(train_data)
    iter=len(FO_time)

    train_loss=np.zeros(iter)
    train_accuracy=np.zeros(iter)
    test_accuracy=np.zeros(iter)
    for i in range(0,iter):
        train_loss[i]=loss_for_D(FO_iter_res[i][0:100],train_data)
        train_accuracy[i]=acc_for_D(FO_iter_res[i][0:100],train_data)
        test_accuracy[i]=acc_for_D(FO_iter_res[i][0:100],test_data)

    p1,=plt.plot(range(0,iter),train_loss)
    plt.xlabel("Number of iterations")
    plt.ylabel("Train loss")
    plt.show()

    p2,=plt.plot(range(0,iter),train_accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Train accuracy")
    plt.show()

    p3,=plt.plot(range(0,iter),test_accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Test accuracy")
    plt.show()

def FO_onedata_run(func,data,x0,w0,lambda_w,lr,iter=100,project=project_simplex):
    lr=np.array(lr)
    FO_iter_res=np.zeros((iter,len(x0)+1))
    FO_time=np.zeros(iter)
    D_x=len(x0)
    D_w=1
    x_opt=x0
    w_opt=1
    best_f=func(x0)
    sigma=1
    flag1=0

    for i in range(0,iter):
        FO_time[i]=time.time()
        FO_iter_res[i][0:D_x]=x_opt
        FO_iter_res[i][D_x:D_x+D_w]=w_opt

        dx=loss_derivative_x_for_D(x_opt,data)
        x_temp=x_opt-dx*lr
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
            print("FO for Min-Max: Iter = %d, lr_x=%f, obj = %3.4f" % (i, lr, y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
        if y_temp<func(x_opt):
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            if flag1%3==0:
                lr=lr*0.98
    return x_opt,FO_iter_res,FO_time

def Average_test(train_data,test_data,x0,iter,alpha,x_gt):
    def loss_Average(x):
        return loss_for_D(x,train_data)

    print("##################################################################")
    print("Average method_test")
    x_opt,Average_iter_res,Average_time=Average_run(loss_Average,x0,step=0.001,lr=alpha,iter=iter)
    np.savez("Average_4_1.npz",x_gt=x_gt,Average_iter_res=Average_iter_res,Average_time=Average_time)

def FO_test(train_data,test_data,x0,iter,alpha,x_gt,lambda_w):
    def loss_FO(x):
        return loss_for_D(x,train_data)

    print("##################################################################")
    print("FO method_test")
    x_opt,FO_iter_res,FO_time=FO_onedata_run(loss_FO,train_data,x0,[1],lambda_w=lambda_w,lr=alpha,iter=iter)
    np.savez("FO_4_1.npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)

def main_test(D_x=100,x_gt0=1,iter=200,alpha=1e-3,beta=1e-3,lambda_w=1e-3,regenerate=False):
    x_gt=x_gt0*np.ones(D_x)
    init_point=np.random.normal(0, 1, D_x+3)

    if regenerate:
        generate_all_dataset(x_gt)#run when x_gt is changed
        save_train_and_test_data()#run when x_gt is changed
        time.sleep(5)

    train_data,test_data=load_train_and_test_data()

    #Average_test(train_data[0],test_data[0],init_point[0:D_x],iter,alpha,x_gt)
    #Average_test_plot(train_data[0],test_data[0],lambda_w=lambda_w,alpha=alpha,beta=beta)
    FO_test(train_data[0],test_data[0],init_point[0:D_x],iter,alpha,x_gt,lambda_w)
    FO_test_plot(train_data[0],test_data[0],lambda_w=lambda_w,alpha=alpha,beta=beta)

if __name__=="__main__":
    main_test(D_x=100,x_gt0=1,iter=500,alpha=3e-4,beta=3e-4,lambda_w=1e-1,regenerate=True)
