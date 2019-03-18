from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
from text_4_1_main import loss_for_D

def worst_loss(iter,iter_res,train_data,test_data,D):
    worst_train_loss=np.zeros(iter)
    worst_test_loss=np.zeros(iter)
    for i in range(0,iter):
        train_worst=-100000
        test_worst=-100000
        for j in range(0,D):
            train_temp=loss_for_D(iter_res[i],train_data[j])
            if train_temp>train_worst:
                train_worst=train_temp
            test_temp=loss_for_D(iter_res[i],test_data[j])
            if test_temp>test_worst:
                test_worst=test_temp
        worst_train_loss[i]=train_worst
        worst_test_loss[i]=test_worst
    return worst_train_loss,worst_test_loss

def AG_train_test_time_plot(train_data,test_data):
    AG=np.load("AG_4_1.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)

    worst_train_loss,worst_test_loss=worst_loss(iter,AG_iter_res,train_data,test_data,D)

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

def train_test_time_plot(train_data,test_data):
    AG=np.load("AG_4_1.npz")
    x_gt=AG['x_gt']
    AG_iter_res=AG['AG_iter_res']
    AG_time=AG['AG_time']
    D=len(train_data)
    iter=len(AG_time)
    worst_train_loss_AG,worst_test_loss_AG=worst_loss(iter,AG_iter_res,train_data,test_data,D)

    Average=np.load("Average_4_1.npz")
    Average_iter_res=Average['Average_iter_res']
    Average_time=Average['Average_time']
    worst_train_loss_Average,worst_test_loss_Average=worst_loss(iter,Average_iter_res,train_data,test_data,D)

    p1,=plt.plot(range(0,iter),worst_train_loss_AG)
    p2,=plt.plot(range(0,iter),worst_train_loss_Average)
    plt.legend([p1, p2], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst train loss")
    plt.show()

    p3,=plt.plot(range(0,iter),worst_test_loss_AG)
    p4,=plt.plot(range(0,iter),worst_test_loss_Average)
    plt.legend([p3, p4], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst test loss")
    plt.show()

    p5,=plt.plot(range(0,iter),AG_time-AG_time[0])
    p6,=plt.plot(range(0,iter),Average_time-Average_time[0])
    plt.legend([p5, p6], ["AG","Average"], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Total time")
    plt.show()