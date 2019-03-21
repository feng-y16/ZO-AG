from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
from text_4_1_AG_GP_utils import*

def worst_loss(iter_res,train_data,test_data):
    iter=np.shape(iter_res)[0]
    D_x=len(train_data[0][0])-1
    iter_res=iter_res[:,0:D_x]
    D_w=len(train_data)
    worst_train_loss=np.zeros(iter)
    worst_test_loss=np.zeros(iter)
    for i in range(0,iter):
        train_worst=-100000
        test_worst=-100000
        for j in range(0,D_w):
            train_temp=loss_for_D(iter_res[i],train_data[j])
            if train_temp>train_worst:
                train_worst=train_temp
            test_temp=loss_for_D(iter_res[i],test_data[j])
            if test_temp>test_worst:
                test_worst=test_temp
        worst_train_loss[i]=train_worst
        worst_test_loss[i]=test_worst
    return worst_train_loss,worst_test_loss

def worst_accuracy(iter_res,train_data,test_data):
    iter=np.shape(iter_res)[0]
    D_x=len(train_data[0][0])-1
    iter_res=iter_res[:,0:D_x]
    D_w=len(train_data)
    worst_train_accuracy=np.zeros(iter)
    worst_test_accuracy=np.zeros(iter)
    for i in range(0,iter):
        train_worst=100000
        test_worst=100000
        for j in range(0,D_w):
            train_temp=acc_for_D(iter_res[i],train_data[j])
            if train_temp<train_worst:
                train_worst=train_temp
            test_temp=acc_for_D(iter_res[i],test_data[j])
            if test_temp<test_worst:
                test_worst=test_temp
        worst_train_accuracy[i]=train_worst
        worst_test_accuracy[i]=test_worst
    return worst_train_accuracy,worst_test_accuracy

def stationary_condition(iter_res,train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    iter=np.shape(iter_res)[0]
    D_w=len(train_data)
    D_x=len(train_data[0][0])-1
    G=np.zeros((iter,D_x+D_w))
    for i in range(0,iter):
        x_opt=iter_res[i][0:D_x]
        w_opt=iter_res[i][D_x:D_x+D_w]
        for j in range(0,D_w):
            G[i][0:D_x]=G[i][0:D_x]+w_opt[j]*loss_derivative_x_for_D(x_opt,train_data[j])
        G[i][D_x:D_x+D_w]=w_opt-project_simplex(w_opt+beta*loss_derivative_w(x_opt,w_opt,lambda_w,train_data))
        G[i][D_x:D_x+D_w]=G[i][D_x:D_x+D_w]/beta
    return np.linalg.norm(G,ord=2,axis=1)

def AG_train_test_time_sc_plot(train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    AG=np.load("AG_4_1.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)

    worst_train_loss,worst_test_loss=worst_loss(AG_iter_res,train_data,test_data)
    worst_train_accuracy,worst_test_accuracy=worst_accuracy(AG_iter_res,train_data,test_data)
    stat_con=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    p1,=plt.plot(range(0,iter),worst_train_loss)
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train loss")
    plt.show()

    p2,=plt.plot(range(0,iter),worst_test_loss)
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test loss")
    plt.show()

    p3,=plt.plot(range(0,iter),AG_time-AG_time[0])
    plt.xlabel("Number of iterations")
    plt.ylabel("Total time")
    plt.show()

    p4,=plt.plot(range(0,iter),stat_con)
    plt.xlabel("Number of iterations")
    plt.ylabel("Stationary condition")
    plt.show()

    p5,=plt.plot(range(0,iter),worst_train_accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train accuracy")
    plt.show()

    p6,=plt.plot(range(0,iter),worst_test_accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test accuracy")
    plt.show()

def AG_Average_train_test_time_sc_plot(train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    AG=np.load("AG_4_1.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)
    worst_train_loss_AG,worst_test_loss_AG=worst_loss(AG_iter_res,train_data,test_data)
    worst_train_accuracy_AG,worst_test_accuracy_AG=worst_accuracy(AG_iter_res,train_data,test_data)
    stat_con_AG=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    Average=np.load("Average_4_1.npz")
    Average_iter_res=Average['Average_iter_res']
    Average_time=Average['Average_time']
    worst_train_loss_Average,worst_test_loss_Average=worst_loss(Average_iter_res,train_data,test_data)
    worst_train_accuracy_Average,worst_test_accuracy_Average=worst_accuracy(Average_iter_res,train_data,test_data)
    stat_con_Average=stationary_condition(Average_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    p11,=plt.plot(range(0,iter),worst_train_loss_AG)
    p12,=plt.plot(range(0,iter),worst_train_loss_Average)
    plt.legend([p11, p12], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train loss")
    plt.show()

    p21,=plt.plot(range(0,iter),worst_test_loss_AG)
    p22,=plt.plot(range(0,iter),worst_test_loss_Average)
    plt.legend([p21, p22], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test loss")
    plt.show()

    p31,=plt.plot(range(0,iter),AG_time-AG_time[0])
    p32,=plt.plot(range(0,iter),Average_time-Average_time[0])
    plt.legend([p31, p32], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Total time")
    plt.show()

    p41,=plt.plot(range(0,iter),stat_con_AG)
    p42,=plt.plot(range(0,iter),stat_con_Average)
    plt.legend([p41, p42], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Stationary condition")
    plt.show()

    p51,=plt.plot(range(0,iter),worst_train_accuracy_AG)
    p52,=plt.plot(range(0,iter),worst_train_accuracy_Average)
    plt.legend([p51, p52], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train accuracy")
    plt.show()

    p61,=plt.plot(range(0,iter),worst_test_accuracy_AG)
    p62,=plt.plot(range(0,iter),worst_test_accuracy_Average)
    plt.legend([p61, p62], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test accuracy")
    plt.show()

def AG_Average_FO_train_test_time_sc_plot(train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    AG=np.load("AG_4_1.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)
    worst_train_loss_AG,worst_test_loss_AG=worst_loss(AG_iter_res,train_data,test_data)
    worst_train_accuracy_AG,worst_test_accuracy_AG=worst_accuracy(AG_iter_res,train_data,test_data)
    stat_con_AG=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    Average=np.load("Average_4_1.npz")
    Average_iter_res=Average['Average_iter_res']
    Average_time=Average['Average_time']
    worst_train_loss_Average,worst_test_loss_Average=worst_loss(Average_iter_res,train_data,test_data)
    worst_train_accuracy_Average,worst_test_accuracy_Average=worst_accuracy(Average_iter_res,train_data,test_data)
    stat_con_Average=stationary_condition(Average_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    FO=np.load("FO_4_1.npz")
    FO_iter_res=FO['FO_iter_res']
    FO_time=FO['FO_time']
    worst_train_loss_FO,worst_test_loss_FO=worst_loss(FO_iter_res,train_data,test_data)
    worst_train_accuracy_FO,worst_test_accuracy_FO=worst_accuracy(FO_iter_res,train_data,test_data)
    stat_con_FO=stationary_condition(FO_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    p11,=plt.plot(range(0,iter),worst_train_loss_AG)
    p12,=plt.plot(range(0,iter),worst_train_loss_Average)
    p13,=plt.plot(range(0,iter),worst_train_loss_FO)
    plt.legend([p11, p12, p13], ["AG","Average","FO"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train loss")
    plt.show()

    p21,=plt.plot(range(0,iter),worst_test_loss_AG)
    p22,=plt.plot(range(0,iter),worst_test_loss_Average)
    p23,=plt.plot(range(0,iter),worst_test_loss_FO)
    plt.legend([p21, p22, p23], ["AG","Average","FO"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test loss")
    plt.show()

    p31,=plt.plot(range(0,iter),AG_time-AG_time[0])
    p32,=plt.plot(range(0,iter),Average_time-Average_time[0])
    p33,=plt.plot(range(0,iter),FO_time-FO_time[0])
    plt.legend([p31, p32, p33], ["AG","Average","FO"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Total time")
    plt.show()

    p41,=plt.plot(range(0,iter),stat_con_AG)
    p42,=plt.plot(range(0,iter),stat_con_Average)
    p43,=plt.plot(range(0,iter),stat_con_FO)
    plt.legend([p41, p42, p43], ["AG","Average","FO"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Stationary condition")
    plt.show()

    p51,=plt.plot(range(0,iter),worst_train_accuracy_AG)
    p52,=plt.plot(range(0,iter),worst_train_accuracy_Average)
    p53,=plt.plot(range(0,iter),worst_train_accuracy_FO)
    plt.legend([p51, p52, p53], ["AG","Average","FO"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train accuracy")
    plt.show()

    p61,=plt.plot(range(0,iter),worst_test_accuracy_AG)
    p62,=plt.plot(range(0,iter),worst_test_accuracy_Average)
    p63,=plt.plot(range(0,iter),worst_test_accuracy_FO)
    plt.legend([p61, p62, p63], ["AG","Average","FO"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test accuracy")
    plt.show()

def train_test_time_sc_plot(train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    AG=np.load("AG_4_1.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)
    worst_train_loss_AG,worst_test_loss_AG=worst_loss(AG_iter_res,train_data,test_data)
    worst_train_accuracy_AG,worst_test_accuracy_AG=worst_accuracy(AG_iter_res,train_data,test_data)
    stat_con_AG=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    Average=np.load("Average_4_1.npz")
    Average_iter_res=Average['Average_iter_res']
    Average_time=Average['Average_time']
    worst_train_loss_Average,worst_test_loss_Average=worst_loss(Average_iter_res,train_data,test_data)
    worst_train_accuracy_Average,worst_test_accuracy_Average=worst_accuracy(Average_iter_res,train_data,test_data)
    stat_con_Average=stationary_condition(Average_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    FO=np.load("FO_4_1.npz")
    FO_iter_res=FO['FO_iter_res']
    FO_time=FO['FO_time']
    worst_train_loss_FO,worst_test_loss_FO=worst_loss(FO_iter_res,train_data,test_data)
    worst_train_accuracy_FO,worst_test_accuracy_FO=worst_accuracy(FO_iter_res,train_data,test_data)
    stat_con_FO=stationary_condition(FO_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    GP=np.load("GP_4_1.npz")
    GP_iter_res=GP['GP_iter_res']
    GP_time=GP['GP_time']
    worst_train_loss_GP,worst_test_loss_GP=worst_loss(GP_iter_res[:,0:len(train_data[0][0])-1],train_data,test_data)
    worst_train_accuracy_GP,worst_test_accuracy_GP=worst_accuracy(GP_iter_res[:,0:len(train_data[0][0])-1],train_data,test_data)
    stat_con_GP=stationary_condition(GP_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    p11,=plt.plot(range(0,iter),worst_train_loss_AG)
    p12,=plt.plot(range(0,iter),worst_train_loss_Average)
    p13,=plt.plot(range(0,iter),worst_train_loss_FO)
    p14,=plt.plot(range(0,iter),worst_train_loss_GP)
    plt.legend([p11, p12, p13, p14], ["AG","Average","FO","GP"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train loss")
    plt.show()

    p21,=plt.plot(range(0,iter),worst_test_loss_AG)
    p22,=plt.plot(range(0,iter),worst_test_loss_Average)
    p23,=plt.plot(range(0,iter),worst_test_loss_FO)
    p24,=plt.plot(range(0,iter),worst_test_loss_GP)
    plt.legend([p21, p22, p23, p24], ["AG","Average","FO","GP"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test loss")
    plt.show()

    plt.yscale('log')
    p31,=plt.plot(range(0,iter),AG_time-AG_time[0])
    p32,=plt.plot(range(0,iter),Average_time-Average_time[0])
    p33,=plt.plot(range(0,iter),FO_time-FO_time[0])
    p34,=plt.plot(range(0,iter),GP_time-GP_time[0])
    plt.legend([p31, p32, p33, p34], ["AG","Average","FO","GP"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Total time")
    plt.show()

    p41,=plt.plot(range(0,iter),stat_con_AG)
    p42,=plt.plot(range(0,iter),stat_con_Average)
    p43,=plt.plot(range(0,iter),stat_con_FO)
    p44,=plt.plot(range(0,iter),stat_con_GP)
    plt.legend([p41, p42, p43, p44], ["AG","Average","FO","GP"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Stationary condition")
    plt.show()

    p51,=plt.plot(range(0,iter),worst_train_accuracy_AG)
    p52,=plt.plot(range(0,iter),worst_train_accuracy_Average)
    p53,=plt.plot(range(0,iter),worst_train_accuracy_FO)
    p54,=plt.plot(range(0,iter),worst_train_accuracy_GP)
    plt.legend([p51, p52, p53], ["AG","Average","FO"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train accuracy")
    plt.show()

    p61,=plt.plot(range(0,iter),worst_test_accuracy_AG)
    p62,=plt.plot(range(0,iter),worst_test_accuracy_Average)
    p63,=plt.plot(range(0,iter),worst_test_accuracy_FO)
    p64,=plt.plot(range(0,iter),worst_test_accuracy_GP)
    plt.legend([p61, p62, p63, p64], ["AG","Average","FO","GP"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test accuracy")
    plt.show()