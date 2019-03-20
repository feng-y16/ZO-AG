from __future__ import print_function
import numpy as np
import random
import math
import time
from text_4_1_AG_GP_utils import *
from text_4_1_dataset import*

class STABLEOPT:
    def __init__(self,train_data,test_data,beta,init_num,mu0,iter=100,step=[0.01,0.01],lr=[0.05,0.05]):
        self.D_x=len(train_data[0][0])-1
        self.D_w=len(train_data)
        self.D=self.D_x+self.D_w
        self.theta0=1
        self.thetai=np.ones(self.D)
        self.sigma2=1
        self.mu0=mu0
        self.init_num=init_num
        self.iter=iter
        self.t=0 
        self.beta=beta
        self.T=len(beta)
        self.GP_time=np.zeros(self.T-self.init_num)
        #self.x=np.zeros((self.T,self.D_x))
        #self.w=np.zeros((self.T,self.D_w))
        self.X=np.zeros((self.T,self.D))
        self.y=np.zeros(self.T)
        self.step=step
        self.lr=lr
        self.iter_initial_point=np.zeros(self.D)
        self.train_data=train_data
        self.test_data=test_data

    def k_(self,x1,x2):
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/(np.max([self.thetai[i]**2, 1e-5]))
        r=np.sqrt(r)  ## distance function
        return self.theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2)

    def k2(self,x1,x2,theta0,thetai):
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/(np.max([thetai[i]**2,1e-5]))
        r=np.sqrt(r)
        return theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2)

    def k_t(self,x):
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k_(x,self.X[i])
        return k

    def k_t2(self,x,theta0,thetai):
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k2(x,self.X[i],theta0,thetai)
        return k

    def K_t(self):
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k_(self.X[i],self.X[j])
        return K

    def K_t2(self,theta0,thetai):
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k2(self.X[i],self.X[j],theta0,thetai)
        return K

    def get_value(self,x,w):
        value=0
        for i in range(0,self.D_w):
            value=value+w[i]*loss_for_D(x,self.train_data[i])
        return value

    def observe(self,x,w):
        #self.x[self.t]=x
        #self.w[self.t]=w
        self.X[self.t][0:self.D_x]=x
        self.X[self.t][self.D_x:self.D_x+self.D_w]=w
        self.y[self.t]=self.get_value(x,w)
        self.t=self.t+1
        return self.y[self.t-1]

    def init(self):
        #self.x[self.t]=np.array(np.random.normal(0, 10, self.D_x))
        #self.w[self.t]=project_simplex(np.array(np.random.normal(0, 10, self.D_w)))
        x=np.array(np.random.normal(0, 10, self.D_x))
        w=project_simplex(np.array(np.random.normal(0, 10, self.D_w)))
        self.X[self.t][0:self.D_x]=x
        self.X[self.t][self.D_x:self.D_x+self.D_w]=w
        self.y[self.t]=self.get_value(x,w)
        self.t=self.t+1
        return 0

    def get_prior(self):#hyper-parameter optimization, no theta0
        m=np.mean(self.y[range(0,self.t)])
        def log_likehood(x): ### maximize the loglikelihood
            thetai=x[range(0,self.D)]
            mu0=x[self.D]    ### prior in mu
            sigma2=x[self.D+1] ### prior in variance
            tempmatrix=self.K_t2(self.theta0,thetai)+sigma2*np.identity(self.t)
            try:
                inv=np.linalg.inv(tempmatrix)
            except:
                print("Singular matrix when computing prior. Small identity added.")
                inv=np.linalg.inv(tempmatrix+0.1*np.identity(self.t))
            finally:
                return 0.5*(((self.y[range(0,self.t)]-mu0).T).dot(inv)).dot(self.y[range(0,self.t)]-mu0)+0.5*math.log(abs(np.linalg.det(tempmatrix)+1e-10))#-0.5*self.t*math.log(2*math.pi)
        bound=np.zeros((self.D+2,2))
        for i in range(0,self.D+2):
            bound[i][0]=-1000000
            bound[i][1]=1000000
        bound[self.D+1][0]=1e-6
        x_opt=ZOSIGNSGD_bounded(log_likehood,np.ones(self.D+2),bound,step=1e-3,lr=1,iter=500,Q=1)
        self.thetai=x_opt[range(0,self.D)]
        self.mu0=x_opt[self.D]
        self.sigma2=x_opt[self.D+1]
        return 0

    def select_init_point(self):
        index=random.randint(0,self.t-1)
        return self.X[index]

    def run_onestep(self,iter):
        if self.t<self.T:
            def ucb(x):
                try:
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
                except:
                    print("Singular matrix when computing UCB. Small identity added.")
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
                finally:
                    mu=self.mu0+((np.array(self.k_t(x)).T).dot(inv).dot(self.y[0:self.t] -self.mu0 ))  ### prior
                    sigma_square=self.theta0**2-(np.array(self.k_t(x)).T).dot(inv).dot(np.array(self.k_t(x)))
                    if(sigma_square<0):
                        print("UCB error: sigma2=",end="")
                        print(sigma_square)
                        print("Let sigma2=0")
                        sigma_square=0
                    return mu+np.sqrt(self.beta[self.t]*sigma_square)
            def lcb(x):
                try:
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
                except:
                    print("Singular matrix when computing LCB. Small identity added.")
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
                finally:
                    mu=self.mu0+((np.array(self.k_t(x)).T).dot(inv).dot(self.y[0:self.t] - self.mu0 ))
                    sigma_square=self.theta0**2-(np.array(self.k_t(x)).T).dot(inv).dot(np.array(self.k_t(x)))
                    if(sigma_square<0):
                        print("LCB error: sigma2=",end="")
                        print(sigma_square)
                        print("Let sigma2=0")
                        sigma_square=0
                return mu-np.sqrt(self.beta[self.t]*sigma_square)
            X_opt=self.select_init_point()
            if self.t==self.init_num:
                self.iter_initial_point=X_opt
            #print(x)

            # ### exhaustive search
            # N_search = 100
            # thr_temp2 = float("-inf")
            # i_search = 0
            # delta_temp = delta + 0
            # for x_search_temp in np.random.uniform(np.array([-0.95, -0.45]),
            #                                        np.array([3.2, 4.4]),
            #                                        (iter, len(x))):
            #     x_search_i = np.resize(x_search_temp, x.shape)
            #     i_search = i_search + 1
            #     def ucb_xfixed(delta):
            #         return ucb(x_search_i+delta)
            #     #### optimize over delta for minimization
            #     # delta_temp = ZOPSGD_bounded_f(ucb_xfixed, delta_temp, distance_fun, self.epsilon, self.step[1], np.zeros(len(x)),
            #     #                          0.1, 50, 2)
            #     thr_temp = float("inf")
            #     for i_search_delta in range(0,N_search):
            #         length = np.random.uniform(0, self.epsilon)
            #         angle = np.pi * np.random.uniform(0, 2)
            #         delta_search_i = np.array([length * np.cos(angle),length * np.sin(angle)])
            #         delta_search_i = np.resize(delta_search_i, delta.shape)
            #
            #         f_val_i = ucb_xfixed(delta_search_i)
            #         if f_val_i < thr_temp:
            #             delta_temp = delta_search_i
            #             thr_temp = f_val_i
            #
            #         if i_search_delta%10 == 0:
            #             print("Search for min ucb_delta: search_time = %d, obj = %3.5f" % (i_search_delta,thr_temp))
            #
            #     f_val_i = ucb_xfixed(delta_temp)
            #
            #     if f_val_i > thr_temp2:
            #         x = x_search_i
            #         delta = delta_temp
            #         thr_temp2 = f_val_i
            #
            #     print("Max-Min-UCB: search_time = %d, obj_max_x = %3.5f" % (i_search, thr_temp2))


            #########original version##############
            x_opt=X_opt[0:self.D_x]
            w_opt=X_opt[self.D_x:self.D_x+self.D_w]
            for i in range(0,iter):
                def ucb_xfixed(w):
                    return ucb(np.concatenate((x_opt,w)))
                #### minimization over ucb for delta
                ### ZOPSGA
                w_opt=ZOPSGA_simplex(ucb_xfixed,w_opt,self.step[1],0.1,1,5) ## 0.1 learning rate
                # ### Exhaustive search
                # thr_temp = float("inf")
                # delta_ex = delta + 0
                # for i_search in range(0,100):
                #     length = np.random.uniform(0, self.epsilon)
                #     angle = np.pi * np.random.uniform(0, 2)
                #     delta_search_i = np.array([length * np.cos(angle),length * np.sin(angle)])
                #     delta_search_i = np.resize(delta_search_i, delta.shape)
                #
                #     f_val_i = ucb_xfixed(delta_search_i)
                #     if f_val_i < thr_temp:
                #         delta_ex = delta_search_i
                #         thr_temp = f_val_i

                # print("ucb_min_delta: iter = %d, obj_min_ZO = %3.5f, obj_min_ex = %3.5f" % (i, ucb_xfixed(delta),ucb_xfixed(delta_ex)))

                #### ucb maximization
                def ucb_deltafixed(x):
                    return ucb(np.concatenate((x,w_opt)))

                ### ZOPSGD
                x_opt=ZOPSGD(ucb_deltafixed,x_opt,self.step[0],0.1,1,5)
                # ### Exhaustive search
                # thr_temp2 = float("-inf")
                # i_search = 0
                # x_ex = x + 0
                # for x_search_temp in np.random.uniform(np.array([-0.95,-0.45]),
                #                                        np.array([3.2, 4.4]),
                #                                      (100, len(x))):
                #     x_search_i = np.resize(x_search_temp, x.shape)
                #     f_val_i = ucb_deltafixed(x_search_i)
                #     i_search = i_search + 1
                #
                #     if f_val_i > thr_temp2:
                #         x_ex = x_search_i
                #         thr_temp2 = f_val_i

                # print("ucb_max_x: iter = %d, obj_max_ZO = %3.5f, obj_max_ex = %3.5f" % (i, ucb_deltafixed(x),ucb_deltafixed(x_ex)))

                if (i%50) == 0:
                    print("ucb_max_x: iter = %d, obj_max_ZO = %3.5f" % (i, ucb_deltafixed(x_opt)))
                    print(x_opt)

            print("Min-LCB: ")
            def lcb_xfixed(w):
                return lcb(np.concatenate((x_opt,w)))
            # delta=ZOPSGD_bounded_f(lcb,x,distance_fun,self.epsilon,self.step[1],np.zeros(len(x)),0.5,100,2) ## self.lr[1]
            w_opt=ZOPSGA_simplex(lcb_xfixed,w_opt,self.step[1],0.1,100,3) ## self.lr[1] ### SL's version
            ### Exhaustive search
            # thr_temp = float("inf")
            # for i_search in range(0, N_search):
            #     # delta_search_tmp = np.random.normal(0, 1, len(delta))
            #     # delta_search_i = self.epsilon * delta_search_tmp / np.linalg.norm(delta_search_tmp)
            #     length = np.random.uniform(0, self.epsilon)
            #     angle = np.pi * np.random.uniform(0, 2)
            #     delta_search_i = np.array([length * np.cos(angle), length * np.sin(angle)])
            #     delta_search_i = np.resize(delta_search_i, delta.shape)
            #
            #     f_val_i = lcb_xfixed(delta_search_i)
            #     if f_val_i < thr_temp:
            #         delta = delta_search_i
            #         thr_temp = f_val_i
            #     if i_search % 10 == 0:
            #         print("Search for min lcb_delta: search_time = %d, obj = %3.5f" % (i_search, thr_temp))

            self.X[self.t][0:self.D_x]=x_opt ### update samples
            self.X[self.t][self.D_x:self.D_x+self.D_w]=w_opt ### update samples
            self.observe(x_opt,w_opt) ### update observations
        else:
            print("t value error!")
            return 0

    def run(self):
        for i in range(0,self.init_num):
            self.init()
        print("Init done")
        for i in range(0,self.T-self.init_num):
            self.GP_time[i]=time.time()
            print(i+1,end="")
            print("/",end="")
            print(self.T-self.init_num)
            if (i%10) == 0:
                print("Getting prior.....")
                self.get_prior()
            else:
                print("Skip updating prior")

            print("theta0=",end="")
            print(self.theta0)
            print("thetai=",end="")
            print(self.thetai)
            print("mu0=",end="")
            print(self.mu0)
            print("sigma2=",end="")
            print(self.sigma2)

            print("Get prior done")
            if self.sigma2<=0:
                print("Prior sigma invaild!")
            self.run_onestep(self.iter)
            print("!!!!!!!!!!!GP max-min iteration %d is done!!!!!!!!!" % i)
        #np.savez('GP_time.npz',GP_time=GP_time)
        print("Done.")
        return 0
    def print(self):
        print("##################################################################")
        print("X:")
        print(self.X)
        print("y:")
        print(self.y)

if __name__=="__main__":
    random.seed(10)
    train_data,test_data=load_train_and_test_data()
    optimizer=STABLEOPT(train_data=train_data,test_data=test_data,beta=4*np.ones(2),init_num=1,mu0=0,iter=50,step=[0.05,0.05],lr=[0.05,0.05])
    optimizer.run()
    optimizer.print()