from __future__ import print_function
import numpy as np
import random

def sigmoid_truncated(x):
    if 1.0/(1+np.exp(-x))>0.5:
        return 1
    else: 
        return 0

def generate_data(n,sigma2,x,noise_std=1e-3):#generate a dataset
    m=len(x)
    data=np.zeros((n,m+1))
    sigma=np.sqrt(sigma2)
    for i in range(0,n):
        a=np.random.normal(0, sigma, m)
        noise=np.random.normal(0, noise_std)
        c=sigmoid_truncated((a.T).dot(x)+noise)
        data[i][0:m]=a
        data[i][m]=c
    return data

def save_data(filename,data):#save a dataset, here filename do not need ".npz"
    np.savez(filename,data=data)

def generate_train_and_test_data(train_ratio,filename,shuffle=False):#load generated data, get trainning and testing data
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

def generate_all_dataset(x=[1,1],filename=["D"],N=[1000],sigma=[5],noise_std=[0.2]):
    data=generate_data(N[0],sigma[0],x,noise_std[0])
    save_data(filename[0],data)

def save_train_and_test_data(shuffle=False,train_ratio=0.7,filename=["D"]):#for generated datasets, divide them into training and testing part, and save
    train_data=[]
    test_data=[]
    for i in range(0,len(filename)):
        temp_train,temp_test=generate_train_and_test_data(train_ratio,filename[i],shuffle=shuffle)
        train_data.append(temp_train)
        test_data.append(temp_test)
    np.savez("train_4_2",train_data=train_data)
    np.savez("test_4_2",test_data=test_data)
    return train_data,test_data

def load_train_and_test_data():#load training and testing data
    train_data=np.load("train_4_2.npz")
    test_data=np.load("test_4_2.npz")
    return train_data['train_data'][0],test_data['test_data'][0]

def generate_index(length,b,iter):
    index=[]
    for i in range(0,iter):
        temp=np.array(random.sample(range(0,length-1),b));
        index.append(temp)
    np.savez("index_4_2",index=index)

def load_index():
    index=np.load("index_4_2.npz")
    index=index['index']
    return index

if __name__=="__main__":
    n=10
    sigma2=1
    x=[1,1]
    filename="D"
    generate_train_and_test_data(0.7,filename,True)