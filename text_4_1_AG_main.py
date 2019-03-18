from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_AG_GP_utils import*
from text_4_1_dataset import*
from text_4_1_plot import *

def loss_for_D(x,data):#计算对于每一个dataset的loss
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=np.log(1+np.exp(-a.dot(x)))
    #loss=-(c.dot(np.log(h))+(1-c).dot(np.log(1-h)))
    loss=-(c.dot(1/h)+(1-c).dot(1-1/h))
    return loss


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

if __name__=="__main__":
    x_gt=[1,1]

    #generate_all_dataset(x_gt)#更改x_gt才需要运行
    #save_train_and_test_data()#更改x_gt才需要运行

    train_data,test_data=load_train_and_test_data()
    #AG_main(train_data,test_data,x_gt=x_gt,lambda_w=1)
    AG_train_test_time_plot(train_data,test_data)