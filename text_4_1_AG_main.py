from __future__ import print_function
import numpy as np
import random
import time
from text_4_1_AG_GP_utils import*
from text_4_1_dataset import*

def loss_for_D(x,data):
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    h=np.log(1+np.exp(-a.dot(x)))
    #loss=-(c.dot(np.log(h))+(1-c).dot(np.log(1-h)))
    loss=-(c.dot(1/h)+(1-c).dot(1-1/h))
    return loss


def main():
    x_gt=[1,1]#
    lambda_w=1#
    generate_all(x_gt)
    train_data,test_data=get_data()
    D_x=len(x_gt)
    D=len(train_data)

    x0=np.zeros(D_x)#
    w0=1.0/D*np.ones(D)#

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
    x_opt,AG_iter_res,AG_time=AG_run(loss_AG,x0,w0,step=[0.01,0.01],lr=[2e-4,2e-4],iter=500,inner_iter=1)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    np.savez("AG_4_1.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)

if __name__=="__main__":
    main()