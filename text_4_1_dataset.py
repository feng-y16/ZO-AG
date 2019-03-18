from __future__ import print_function
import numpy as np
import random

def sign(x):
    if x<0:
        return 0
    else: 
        return 1

def generate_data(n,sigma2,x):
    m=len(x)
    data=np.zeros((n,m+1))
    sigma=np.sqrt(sigma2)
    for i in range(0,n):
        a=np.random.normal(0, sigma, m)
        noise=np.random.normal(0, 1e-3)
        c=sign((a.T).dot(x)+noise)
        data[i][0:m]=a
        data[i][m]=c
    return data

def save_data(filename,data):#filename不需要加.npz
    np.savez(filename,data=data)

def generate_original_data(n,sigma2,x,filename):
    data=generate_data(n,sigma2,x)
    save_data(filename,data)

def generate_train_and_test_data(train_ratio,filename,shuffle=False):
    data=np.load(filename+".npz")
    data=data['data']
    n=np.shape(data)[0]
    train_n=int(np.round(train_ratio*n))
    test_n=n-train_n
    if shuffle:
        shuffled_indexes=list(range(0,n))
        random.shuffle(shuffled_indexes)
        data=data[shuffled_indexes]
    train_data=data[0:train_n]
    test_data=data[train_n:n]
    #print(train_data)
    #print(test_data)
    return train_data,test_data

def generate_all(x=[1,1],filename=["D1","D2","D3"],N=[800,1000,500],sigma2=[1,10,0.1]):
    for i in range(0,len(filename)):
        generate_original_data(N[i],sigma2[i],x,filename[i])

def get_data(shuffle=False,train_ratio=0.7,filename=["D1","D2","D3"]):
    train_data=[]
    test_data=[]
    for i in range(0,len(filename)):
        temp_train,temp_test=generate_train_and_test_data(train_ratio,filename[i],shuffle=shuffle)
        train_data.append(temp_train)
        test_data.append(temp_test)
    #print(train_data)
    #print(test_data)
    return train_data,test_data

if __name__=="__main__":
    #n=10
    #sigma2=1
    #x=[1,1]
    #filename="D1"
    #generate_original_data(n,sigma2,x,filename)
    #generate_train_and_test_data(0.7,filename,True)


    generate_all()
    get_data()