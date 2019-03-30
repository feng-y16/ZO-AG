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

def all_loss(iter_res,train_data,test_data):
    iter=np.shape(iter_res)[0]
    D_x=len(train_data[0][0])-1
    iter_res=iter_res[:,0:D_x]
    D_w=len(train_data)
    all_train_loss=np.zeros((D_w,iter))
    all_test_loss=np.zeros((D_w,iter))
    for i in range(0,iter):
        train_worst=-100000
        test_worst=-100000
        for j in range(0,D_w):
            all_train_loss[j][i]=loss_for_D(iter_res[i],train_data[j])
            all_test_loss[j][i]=loss_for_D(iter_res[i],test_data[j])
    return all_train_loss,all_test_loss

def all_acc(iter_res,train_data,test_data):
    iter=np.shape(iter_res)[0]
    D_x=len(train_data[0][0])-1
    iter_res=iter_res[:,0:D_x]
    D_w=len(train_data)
    all_train_acc=np.zeros((D_w,iter))
    all_test_acc=np.zeros((D_w,iter))
    for i in range(0,iter):
        train_worst=-100000
        test_worst=-100000
        for j in range(0,D_w):
            all_train_acc[j][i]=acc_for_D(iter_res[i],train_data[j])
            all_test_acc[j][i]=acc_for_D(iter_res[i],test_data[j])
    return all_train_acc,all_test_acc

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

def plot_nineline(data1,data2,data3,xlabel,ylabel,legend=["AG","Average","FO"],loc='upper left',filename=None):
    iter=np.shape(data1)[1]
    p11,=plt.plot(range(0,iter),data1[0],color='red',linestyle='-.')
    p12,=plt.plot(range(0,iter),data2[0],color='green',linestyle='-.')
    p13,=plt.plot(range(0,iter),data3[0],color='blue',linestyle='-.')
    p21,=plt.plot(range(0,iter),data1[1],color='red',linestyle=':')
    p22,=plt.plot(range(0,iter),data2[1],color='green',linestyle=':')
    p23,=plt.plot(range(0,iter),data3[1],color='blue',linestyle=':')
    p31,=plt.plot(range(0,iter),data1[2],color='red',linestyle='--')
    p32,=plt.plot(range(0,iter),data2[2],color='green',linestyle='--')
    p33,=plt.plot(range(0,iter),data3[2],color='blue',linestyle='--')
    plt.legend([p11, p12, p13, p21, p22, p23, p31, p32, p33], [legend[0]+"_D1",legend[1]+"_D1",legend[2]+"_D1",
                                                               legend[0]+"_D2",legend[1]+"_D2",legend[2]+"_D2",legend[0]+"_D3",legend[1]+"_D3",legend[2]+"_D3"], loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename!=None:
        plt.savefig(filename)
    plt.show()

def plot_threeline(data1,data2,data3,xlabel,ylabel,legend=["AG","Average","FO"],loc='upper left',filename=None):
    iter=np.shape(data1)[0]
    p1,=plt.plot(range(0,iter),data1)
    p2,=plt.plot(range(0,iter),data2)
    p3,=plt.plot(range(0,iter),data3)
    plt.legend([p1, p2, p3], legend, loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename!=None:
        plt.savefig(filename)
    plt.show()

def AG_Average_FO_train_test_time_sc_plot_all(train_data,test_data,lambda_w=1,alpha=1e-4,beta=1e-4):
    AG=np.load("AG_4_1.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)
    worst_train_loss_AG,worst_test_loss_AG=worst_loss(AG_iter_res,train_data,test_data)
    worst_train_accuracy_AG,worst_test_accuracy_AG=worst_accuracy(AG_iter_res,train_data,test_data)
    all_train_loss_AG,all_test_loss_AG=all_loss(AG_iter_res,train_data,test_data)
    all_train_accuracy_AG,all_test_accuracy_AG=all_acc(AG_iter_res,train_data,test_data)
    stat_con_AG=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    Average=np.load("Average_4_1.npz")
    Average_iter_res=Average['Average_iter_res']
    Average_time=Average['Average_time']
    worst_train_loss_Average,worst_test_loss_Average=worst_loss(Average_iter_res,train_data,test_data)
    worst_train_accuracy_Average,worst_test_accuracy_Average=worst_accuracy(Average_iter_res,train_data,test_data)
    all_train_loss_Average,all_test_loss_Average=all_loss(Average_iter_res,train_data,test_data)
    all_train_accuracy_Average,all_test_accuracy_Average=all_acc(Average_iter_res,train_data,test_data)
    stat_con_Average=stationary_condition(Average_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    FO=np.load("FO_4_1.npz")
    FO_iter_res=FO['FO_iter_res']
    FO_time=FO['FO_time']
    worst_train_loss_FO,worst_test_loss_FO=worst_loss(FO_iter_res,train_data,test_data)
    worst_train_accuracy_FO,worst_test_accuracy_FO=worst_accuracy(FO_iter_res,train_data,test_data)
    all_train_loss_FO,all_test_loss_FO=all_loss(FO_iter_res,train_data,test_data)
    all_train_accuracy_FO,all_test_accuracy_FO=all_acc(FO_iter_res,train_data,test_data)
    stat_con_FO=stationary_condition(FO_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    plot_threeline(worst_train_loss_AG,worst_train_loss_Average,worst_train_loss_FO,
                   "Number of iterations","Worst train loss",legend=["AG","Average","FO"],loc='upper left',filename="worst_train_loss.png")
    #plot_threeline(worst_train_accuracy_AG,worst_train_accuracy_Average,worst_train_accuracy_FO,
                   #"Number of iterations","Worst train accuracy",legend=["AG","Average","FO"],loc='upper left',filename="worst_train_accuracy.png")
    plot_threeline(worst_test_accuracy_AG,worst_test_accuracy_Average,worst_test_accuracy_FO,
                   "Number of iterations","Worst test accuracy",legend=["AG","Average","FO"],loc='upper left',filename="worst_test_accuracy.png")
    plot_threeline(stat_con_AG,stat_con_Average,stat_con_FO,
                   "Number of iterations","Stationary condition",legend=["AG","Average","FO"],loc='upper left',filename="stationary_condition.png")
    plot_threeline(AG_time-AG_time[0],Average_time-Average_time[0],FO_time-FO_time[0],
                   "Number of iterations","Total time",legend=["AG","Average","FO"],loc='upper left',filename="time.png")
    plot_nineline(all_train_loss_AG,all_train_loss_Average,all_train_loss_FO,
                   "Number of iterations","Train loss",legend=["AG","Average","FO"],loc='upper left',filename="train_loss.png")
    plot_nineline(all_test_accuracy_AG,all_test_accuracy_Average,all_test_accuracy_FO,
                   "Number of iterations","Test accuracy",legend=["AG","Average","FO"],loc='upper left',filename="test_accuracy.png")

def plot_shaded(mean,std):
    iter=len(mean)
    mean=np.array(mean)
    std=np.array(std)
    low=mean-std
    high=mean+std
    p1,=plt.plot(range(0,iter),mean)
    plt.fill_between(range(0,iter),low,high,alpha=0.5)
    return p1

def plot_threeline_shaded(data1_mean,data1_std,data2_mean,data2_std,data3_mean,data3_std,xlabel,ylabel,legend=["AG","Average","FO"],loc='upper left',filename=None):
    iter=np.shape(data1_mean)[0]
    p1=plot_shaded(data1_mean,data1_std)
    p2=plot_shaded(data2_mean,data2_std)
    p3=plot_shaded(data3_mean,data3_std)
    plt.legend([p1, p2, p3], legend, loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename!=None:
        plt.savefig(filename)
    plt.show()

def mean_std(data):
    n=len(data)
    iter=len(data[0])
    mean=np.zeros(iter)
    std=np.zeros(iter)
    for i in range(0,iter):
        iter_data=np.zeros(n)
        for j in range(0,n):
            iter_data[j]=np.array(data[j][i])
        mean[i]=np.mean(iter_data)
        std[i]=np.std(iter_data)
    return mean,std

def multiplot_all(train_data,test_data,lambda_w,alpha,beta,times):
    wtrl_AG=[]
    wtel_AG=[]
    wtrc_AG=[]
    wtec_AG=[]
    atrl_AG=[]
    atel_AG=[]
    sc_AG=[]
    wtrl_Average=[]
    wtel_Average=[]
    wtrc_Average=[]
    wtec_Average=[]
    atrl_Average=[]
    atel_Average=[]
    sc_Average=[]
    wtrl_FO=[]
    wtel_FO=[]
    wtrc_FO=[]
    wtec_FO=[]
    atrl_FO=[]
    atel_FO=[]
    sc_FO=[]
    for i in range(0,times):
        filename=str(i)
        AG=np.load("AG_4_1_"+filename+".npz")
        x_gt=AG['x_gt']
        AG_iter_res=AG['AG_iter_res']
        AG_time=AG['AG_time']
        D=len(train_data)
        iter=len(AG_time)
        worst_train_loss_AG,worst_test_loss_AG=worst_loss(AG_iter_res,train_data,test_data)
        worst_train_accuracy_AG,worst_test_accuracy_AG=worst_accuracy(AG_iter_res,train_data,test_data)
        all_train_loss_AG,all_test_loss_AG=all_loss(AG_iter_res,train_data,test_data)
        all_train_accuracy_AG,all_test_accuracy_AG=all_acc(AG_iter_res,train_data,test_data)
        stat_con_AG=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)
        wtrl_AG.append(worst_train_loss_AG)
        wtel_AG.append(worst_test_loss_AG)
        atrl_AG.append(all_train_loss_AG)
        atel_AG.append(all_test_loss_AG)
        wtrc_AG.append(worst_train_accuracy_AG)
        wtec_AG.append(worst_test_accuracy_AG)
        sc_AG.append(stat_con_AG)

        Average=np.load("Average_4_1_"+filename+".npz")
        Average_iter_res=Average['Average_iter_res']
        Average_time=Average['Average_time']
        worst_train_loss_Average,worst_test_loss_Average=worst_loss(Average_iter_res,train_data,test_data)
        worst_train_accuracy_Average,worst_test_accuracy_Average=worst_accuracy(Average_iter_res,train_data,test_data)
        all_train_loss_Average,all_test_loss_Average=all_loss(Average_iter_res,train_data,test_data)
        all_train_accuracy_Average,all_test_accuracy_Average=all_acc(Average_iter_res,train_data,test_data)
        stat_con_Average=stationary_condition(Average_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)
        wtrl_Average.append(worst_train_loss_Average)
        wtel_Average.append(worst_test_loss_Average)
        atrl_Average.append(all_train_loss_Average)
        atel_Average.append(all_test_loss_Average)
        wtrc_Average.append(worst_train_accuracy_Average)
        wtec_Average.append(worst_test_accuracy_Average)
        sc_Average.append(stat_con_Average)

        FO=np.load("FO_4_1_"+filename+".npz")
        FO_iter_res=FO['FO_iter_res']
        FO_time=FO['FO_time']
        worst_train_loss_FO,worst_test_loss_FO=worst_loss(FO_iter_res,train_data,test_data)
        worst_train_accuracy_FO,worst_test_accuracy_FO=worst_accuracy(FO_iter_res,train_data,test_data)
        all_train_loss_FO,all_test_loss_FO=all_loss(FO_iter_res,train_data,test_data)
        all_train_accuracy_FO,all_test_accuracy_FO=all_acc(FO_iter_res,train_data,test_data)
        stat_con_FO=stationary_condition(FO_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)
        wtrl_FO.append(worst_train_loss_FO)
        wtel_FO.append(worst_test_loss_FO)
        atrl_FO.append(all_train_loss_FO)
        atel_FO.append(all_test_loss_FO)
        wtrc_FO.append(worst_train_accuracy_FO)
        wtec_FO.append(worst_test_accuracy_FO)
        sc_FO.append(stat_con_FO)

    wtrl_AG_mean,wtrl_AG_std=mean_std(wtrl_AG)
    wtel_AG_mean,wtel_AG_std=mean_std(wtel_AG)
    #atrl_AG_mean,atrl_AG_std=mean_std(atrl_AG)
    wtrc_AG_mean,wtrc_AG_std=mean_std(wtrc_AG)
    wtec_AG_mean,wtec_AG_std=mean_std(wtec_AG)
    sc_AG_mean,sc_AG_std=mean_std(sc_AG)

    wtrl_Average_mean,wtrl_Average_std=mean_std(wtrl_Average)
    wtel_Average_mean,wtel_Average_std=mean_std(wtel_Average)
    #atrl_Average_mean,atrl_Average_std=mean_std(atrl_Average)
    wtrc_Average_mean,wtrc_Average_std=mean_std(wtrc_Average)
    wtec_Average_mean,wtec_Average_std=mean_std(wtec_Average)
    sc_Average_mean,sc_Average_std=mean_std(sc_Average)

    wtrl_FO_mean,wtrl_FO_std=mean_std(wtrl_FO)
    wtel_FO_mean,wtel_FO_std=mean_std(wtel_FO)
    #atrl_FO_mean,atrl_FO_std=mean_std(atrl_FO)
    wtrc_FO_mean,wtrc_FO_std=mean_std(wtrc_FO)
    wtec_FO_mean,wtec_FO_std=mean_std(wtec_FO)
    sc_FO_mean,sc_FO_std=mean_std(sc_FO)

    plot_threeline_shaded(wtrl_AG_mean,wtrl_AG_std,wtrl_Average_mean,wtrl_Average_std,wtrl_FO_mean,wtrl_FO_std,
                   "Number of iterations","Worst train loss",legend=["AG","Average","FO"],loc='upper left',filename="worst_train_loss_shaded.png")
    plot_threeline_shaded(wtrc_AG_mean,wtrc_AG_std,wtrc_Average_mean,wtrc_Average_std,wtrc_FO_mean,wtrc_FO_std,
                   "Number of iterations","Worst train accuracy",legend=["AG","Average","FO"],loc='upper left',filename="worst_train_accuracy_shaded.png")
    plot_threeline_shaded(wtec_AG_mean,wtec_AG_std,wtec_Average_mean,wtec_Average_std,wtec_FO_mean,wtec_FO_std,
                   "Number of iterations","Worst test accuracy",legend=["AG","Average","FO"],loc='upper left',filename="worst_test_accuracy_shaded.png")
    plot_threeline_shaded(sc_AG_mean,sc_AG_std,sc_Average_mean,sc_Average_std,sc_FO_mean,sc_FO_std,
                   "Number of iterations","Stationary condition",legend=["AG","Average","FO"],loc='upper left',filename="stationary_condition_shaded.png")