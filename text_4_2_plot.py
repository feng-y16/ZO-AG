from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
from text_4_2_AG_GP_utils import*

def all_loss(iter_res,lambda_x,train_data,test_data):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_loss=np.zeros(iter)
    all_test_loss=np.zeros(iter)
    for i in range(0,iter):
        all_train_loss[i]=loss_function(iter_res_delta[i],iter_res_x[i],lambda_x,train_data)
        all_test_loss[i]=loss_function(iter_res_delta[i],iter_res_x[i],lambda_x,test_data)
    return all_train_loss,all_test_loss

def all_acc(iter_res,lambda_x,train_data,test_data):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_acc=np.zeros(iter)
    all_test_acc=np.zeros(iter)
    for i in range(0,iter):
        all_train_acc[i]=acc_for_D(iter_res_delta[i],iter_res_x[i],lambda_x,train_data)
        all_test_acc[i]=acc_for_D(iter_res_delta[i],iter_res_x[i],lambda_x,test_data)
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

def plot_twoline(data1,data2,xlabel,ylabel,legend=["AG","FO"],loc='upper left',filename=None):
    iter=np.shape(data1)[0]
    p1,=plt.plot(range(0,iter),data1)
    p2,=plt.plot(range(0,iter),data2)
    plt.legend([p1, p2], legend, loc='upper left')
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

def AG_FO_plot_all(train_data,test_data,lambda_x=1,alpha=1e-4,beta=1e-4):
    AG=np.load("AG_4_2.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)
    all_train_loss_AG,all_test_loss_AG=all_loss(AG_iter_res,lambda_x,train_data,test_data)
    all_train_accuracy_AG,all_test_accuracy_AG=all_acc(AG_iter_res,lambda_x,train_data,test_data)
    #stat_con_AG=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    FO=np.load("FO_4_2.npz")
    FO_iter_res=FO['FO_iter_res']
    FO_time=FO['FO_time']
    all_train_loss_FO,all_test_loss_FO=all_loss(FO_iter_res,lambda_x,train_data,test_data)
    all_train_accuracy_FO,all_test_accuracy_FO=all_acc(FO_iter_res,lambda_x,train_data,test_data)
    #stat_con_FO=stationary_condition(FO_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)

    plot_twoline(all_train_loss_AG,all_train_loss_FO,
                   "Number of iterations","Train loss",legend=["AG","FO"],loc='upper left',filename="train_loss.png")
    plot_twoline(all_test_accuracy_AG,all_test_accuracy_FO,
                   "Number of iterations","Worst test accuracy",legend=["AG","FO"],loc='upper left',filename="test_accuracy.png")
    #plot_threeline(stat_con_AG,stat_con_Average,stat_con_FO,
                   #"Number of iterations","Stationary condition",legend=["AG","Average","FO"],loc='upper left',filename="stationary_condition.png")
    plot_twoline(AG_time-AG_time[0],FO_time-FO_time[0],
                   "Number of iterations","Total time",legend=["AG","FO"],loc='upper left',filename="time.png")

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

def plot_twoline_shaded(data1_mean,data1_std,data2_mean,data2_std,xlabel,ylabel,legend=["AG","FO"],loc='upper left',filename=None):
    iter=np.shape(data1_mean)[0]
    p1=plot_shaded(data1_mean,data1_std)
    p2=plot_shaded(data2_mean,data2_std)
    plt.legend([p1, p2], legend, loc='upper left')
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

def mean_all(data):
    times=len(data)
    n=np.shape(data[0])[0]
    iter=np.shape(data[0])[1]
    mean=np.zeros((n,iter))
    std=np.zeros((n,iter))
    mean_mean=np.zeros(iter)
    mean_std=np.zeros(iter)
    for i in range(0,iter):
        temp_data1=np.zeros((n,times))
        for j in range(0,n):
            temp_data2=np.zeros(times)
            for k in range(0,times):
                temp_data2[k]=data[k][j][i]
                temp_data1[j][k]=data[k][j][i]
            mean[j][i]=np.mean(temp_data2)
            std[j][i]=np.std(temp_data2)
        temp_data1=temp_data1.flatten()
        mean_mean[i]=np.mean(temp_data1)
        mean_std[i]=np.std(temp_data1)
    return mean,std,mean_mean,mean_std

def mean(data):
    n=len(data)
    iter=len(data[0])
    mean=np.zeros(iter)
    for i in range(0,iter):
        iter_data=np.zeros(n)
        for j in range(0,n):
            iter_data[j]=np.array(data[j][i])
        mean[i]=np.mean(iter_data)
    return mean

def multiplot_all(train_data,test_data,lambda_x,alpha,beta,times):
    wtrl_AG=[]
    wtel_AG=[]
    wtrc_AG=[]
    wtec_AG=[]
    atrl_AG=[]
    atel_AG=[]
    atrc_AG=[]
    atec_AG=[]
    sc_AG=[]
    wtrl_FO=[]
    wtel_FO=[]
    wtrc_FO=[]
    wtec_FO=[]
    atrl_FO=[]
    atel_FO=[]
    atrc_FO=[]
    atec_FO=[]
    sc_FO=[]
    for i in range(0,times):
        filename=str(i)
        AG=np.load("AG_4_2_"+filename+".npz")
        x_gt=AG['x_gt']
        AG_iter_res=AG['AG_iter_res']
        AG_time=AG['AG_time']
        D=len(train_data)
        iter=len(AG_time)
        all_train_loss_AG,all_test_loss_AG=all_loss(AG_iter_res,lambda_x,train_data,test_data)
        all_train_accuracy_AG,all_test_accuracy_AG=all_acc(AG_iter_res,lambda_x,train_data,test_data)
        #stat_con_AG=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)
        atrl_AG.append(all_train_loss_AG)
        atel_AG.append(all_test_loss_AG)
        atrc_AG.append(all_train_accuracy_AG)
        atec_AG.append(all_test_accuracy_AG)
        #sc_AG.append(stat_con_AG)

        FO=np.load("FO_4_2_"+filename+".npz")
        FO_iter_res=FO['FO_iter_res']
        FO_time=FO['FO_time']
        all_train_loss_FO,all_test_loss_FO=all_loss(FO_iter_res,lambda_x,train_data,test_data)
        all_train_accuracy_FO,all_test_accuracy_FO=all_acc(FO_iter_res,lambda_x,train_data,test_data)
        #stat_con_FO=stationary_condition(FO_iter_res,train_data,test_data,lambda_w=lambda_w,alpha=alpha,beta=beta)
        atrl_FO.append(all_train_loss_FO)
        atel_FO.append(all_test_loss_FO)
        atrc_FO.append(all_train_accuracy_FO)
        atec_FO.append(all_test_accuracy_FO)
        #sc_FO.append(stat_con_FO)

    atrl_AG_mean,atrl_AG_std=mean_std(atrl_AG)
    atel_AG_mean,atel_AG_std=mean_std(atel_AG)
    atrc_AG_mean,atrc_AG_std=mean_std(atrc_AG)
    atec_AG_mean,atec_AG_std=mean_std(atec_AG)
    #sc_AG_mean,sc_AG_std=mean_std(sc_AG)

    atrl_FO_mean,atrl_FO_std=mean_std(atrl_FO)
    atel_FO_mean,atel_FO_std=mean_std(atel_FO)
    atrc_FO_mean,atrc_FO_std=mean_std(atrc_FO)
    atec_FO_mean,atec_FO_std=mean_std(atec_FO)
    #sc_FO_mean,sc_FO_std=mean_std(sc_FO)

    plot_twoline_shaded(atrl_AG_mean,atrl_AG_std,atrl_FO_mean,atrl_FO_std,
                   "Number of iterations","Train loss",legend=["AG","FO"],loc='upper left',filename="train_loss_shaded.png")
    plot_twoline_shaded(atec_AG_mean,atec_AG_std,atec_FO_mean,atec_FO_std,
                   "Number of iterations","Test accuracy",legend=["AG","FO"],loc='upper left',filename="test_accuracy_shaded.png")
    #plot_threeline_shaded(sc_AG_mean,sc_AG_std,sc_Average_mean,sc_Average_std,sc_FO_mean,sc_FO_std,
    #               "Number of iterations","Stationary condition",legend=["AG","Average","FO"],loc='upper left',filename="stationary_condition_shaded.png")

def lambda_plot(data,lambda_x,ylabel,xlabel="Number of iterations",filename=None):
    iter=np.shape(data[0])[0]
    legend=[]
    for i in range(0,len(lambda_x)):
        plt.plot(range(0,iter),data[i])
        legend.append("lambda="+str(lambda_x[i]))
    plt.legend(legend,loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename!=None:
        plt.savefig(filename)
    plt.show()

def multilambda_plot_all(train_data,test_data,lambda_x,alpha,beta,times):
    ATRL=[]
    ATEC=[]
    #SC=[]
    for i in range(0,len(lambda_x)):
        atrl_AG=[]
        atec_AG=[]
        #sc_AG=[]
        lambda_i=lambda_x[i]
        for j in range(0,times):
            AG=np.load("AG_4_2_"+"lambda_"+str(lambda_i)+"_"+str(j)+".npz")
            x_gt=AG['x_gt']
            AG_iter_res=AG['AG_iter_res']
            AG_time=AG['AG_time']
            D=len(train_data)
            iter=len(AG_time)
            atrl,atel=all_loss(AG_iter_res,lambda_i,train_data,test_data)
            atrc,atec=all_acc(AG_iter_res,lambda_i,train_data,test_data)
            #sc=stationary_condition(AG_iter_res,train_data,test_data,lambda_w=lambda_i,alpha=alpha,beta=beta)
            atrl_AG.append(atrl)
            atec_AG.append(atec)
            #sc_AG.append(sc)
        atrl=mean(atrl_AG)
        atec=mean(atec_AG)
        #sc=mean(sc_AG)
        ATRL.append(atrl)
        ATEC.append(atec)
        #SC.append(sc)
    lambda_plot(ATRL,lambda_x,"Train loss","Number of iterations","lambda_train_loss.png")
    lambda_plot(ATEC,lambda_x,"Test accuracy","Number of iterations","lambda_test_accuracy.png")
    #lambda_plot(SC,lambda_x,"Stationary condition","Number of iterations","lambda_stationary_condition.png")
