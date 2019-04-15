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

def generate_original_data(n,sigma2,x,filename,noise_std=1e-3):#generate+save
    data=generate_data(n,sigma2,x,noise_std=noise_std)
    save_data(filename,data)

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

def generate_all_dataset(x=[1,1],filename=["D1","D2","D3"],N=[800,1000,500],sigma2=[1,10,0.1]):#generated datasets and save
    for i in range(0,len(filename)):
        generate_original_data(N[i],sigma2[i],x,filename[i])

def generate_all_dataset_heter_var(x=[1,1],filename=["D1","D2","D3"],N=[500,500,500,500,500,500],sigma=[1,7,5,5,0.1,10]):
    for i in range(0,3):
        data1=generate_data(N[2*i],sigma[2*i],x)
        data2=generate_data(N[2*i+1],sigma[2*i+1],x)
        data=np.concatenate((data1,data2))
        save_data(filename[i],data)

#def generate_all_dataset_heter_mean(x=[1,1],filename=["D1","D2","D3"],N=[1000,1000,1000],sigma=[20,1,1],multiplier=0.5,noise_std=[5,0.1,0.1]):
#    data=generate_data(N[0],sigma[0],x,noise_std[0])
#    save_data(filename[0],data)
#    for i in range(1,3):
#        x_temp=x
#        x_temp=np.array(x_temp)
#        x_temp=x_temp+np.random.uniform(-multiplier,multiplier,len(x))
#        data=generate_data(N[i],sigma[i],x_temp,noise_std[i])
#        save_data(filename[i],data)

def generate_all_dataset_heter_mean(x=[1,1],filename=["D1","D2","D3"],N=[1000,1000,1000],sigma=[5,2,2],multiplier=0,noise_std=[0.2,0.1,0.1]):
    data=generate_data(N[0],sigma[0],x,noise_std[0])
    save_data(filename[0],data)
    for i in range(1,3):
        x_temp=x
        x_temp=np.array(x_temp)
        x_temp=x_temp+np.random.uniform(-multiplier,multiplier,len(x))
        data=generate_data(N[i],sigma[i],x_temp,noise_std[i])
        save_data(filename[i],data)

def save_train_and_test_data(shuffle=False,train_ratio=0.7,filename=["D1","D2","D3"]):#for generated datasets, divide them into training and testing part, and save
    train_data=[]
    test_data=[]
    for i in range(0,len(filename)):
        temp_train,temp_test=generate_train_and_test_data(train_ratio,filename[i],shuffle=shuffle)
        train_data.append(temp_train)
        test_data.append(temp_test)
    np.savez("train_4_1",train_data=train_data)
    np.savez("test_4_1",test_data=test_data)
    return train_data,test_data

def load_train_and_test_data():#load training and testing data
    train_data=np.load("train_4_1.npz")
    test_data=np.load("test_4_1.npz")
    return train_data['train_data'],test_data['test_data']

def generate_index(length,b,iter,num_of_dataset):
    index=[]
    for i in range(0,iter):
        temp=np.zeros((num_of_dataset,b))
        for j in range(0,num_of_dataset):
            temp[j]=np.array(random.sample(range(0,length-1),b));
        index.append(temp)
    np.savez("index_4_1",index=index)

def load_index():
    index=np.load("index_4_1.npz")
    index=index['index']
    return index

if __name__=="__main__":
    n=10
    sigma2=1
    x=[1,1]
    filename="D1"
    generate_original_data(n,sigma2,x,filename)
    generate_train_and_test_data(0.7,filename,True)