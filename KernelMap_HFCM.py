import numpy as np
import matplotlib.pyplot as plt
from FCMs import transferFunc, reverseFunc
from rcga_heuristic_xover import Evolution
import pandas as pd
import time
import random
from pandas import DataFrame
import os

def revise(data):
    column,row =data.shape
    tempdata = data.copy()
    for i in range(column):
        for j in range(row):
            if tempdata[i,j]>1:
                tempdata[i,j]=1
            if tempdata[i,j]<0:
                tempdata[i,j]= 0
    return tempdata

def splitData(dataset, ratio=0.85):
    len_train_data = int(len(dataset) * ratio)
    return dataset[:len_train_data], dataset[len_train_data:]


# form feature matrix from sequence
def create_dataset(seq, belta, Order, current_node):
    Nc, K = seq.shape
    samples = np.zeros(shape=(K, Order * Nc + 2))
    for m in range(Order, K):
        for n_idx in range(Nc):
            for order in range(Order):
                samples[m - Order, n_idx * Order + order + 1] = seq[n_idx, m - 1 - order]
        samples[m - Order, 0] = 1
        samples[m - Order, -1] = reverseFunc(seq[current_node, m], belta, '01')
    return samples


def predict(samples, weight, steepness, belta):
    # samples: each row is a sample, each column is one feature
    K, _ = samples.shape
    predicted_data = np.zeros(shape=(1, K))
    for t in range(K):
        features = samples[t, :-1]
        predicted_data[0, t] = transferFunc(steepness * np.dot(weight, features), belta, '01')
    return predicted_data

# normalize data set into [0, 1] or [-1, 1]
def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:   # 2-D
        N , K = data.shape
        minV = np.zeros(shape=K)
        maxV = np.zeros(shape=K)
        for i in range(N):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':   # normalize to [0, 1]
                    # data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                    data[i, :] =(0.9 - 0.1) * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) + 0.1
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data, maxV, minV
    else:   # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                # data = (0.95 - 0.05) * (data - minV) / (maxV - minV) + 0.05
                data = (0.9 - 0.1) * (data - minV) / (maxV - minV) + 0.1
                # data = (data - minV) / (maxV - minV) 
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data, maxV, minV


# re-normalize data set from [0, 1] or [-1, 1] into its true dimension
def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        Nc, K = data.shape
        for i in range(Nc):
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':   # normalize to [0, 1]
                    data[i, :] = (data[i, :] - 0.1) * (maxV[i] - minV[i]) / (0.9 - 0.1) + minV[i]
                    
                else:
                    data[i, :] = (data[i, :] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
    else:  # 1-D
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                # data = data * (maxV - minV) + minV
                # data = (data - 0.05) * (maxV - minV) / (0.95 - 0.05) + minV
                data = (data - 0.1) * (maxV - minV) / (0.9 - 0.1) + minV
            else:
                data = (data + 1) * (maxV - minV) / 2 + minV
    return data

def random_deposition(X, n = 2):
    N = len(X)
#    Fx1 = np.zeros([n, N])
#    Fx2 = np.zeros([n, N])
#    Fx3 = np.zeros([n, N])
#    Fx4 = np.zeros([n, N])
#    for i in range(n):
#        Fx1[i, :] = np.power(X, i+2)
#        Fx2[i, :] = 1 - Fx1[i, :]
#        Fx3[i, :] = np.power(X, 1/(i+2))
#        Fx4[i, :] = 1- Fx3[i, :]
#    Y = np.vstack((X, Fx1))    
#    Y = np.vstack((Y, Fx2))
#    Y = np.vstack((Y, Fx3))
#    Y = np.vstack((Y, Fx4))
#    Y = np.vstack((Y, 1-X))
    Fx1 = np.zeros([n, N])
    Fx3 = np.zeros([n, N])
    Fx2 = np.zeros([n, N])
    Fx4 = np.zeros([n, N])
    for i in range(n):
        Fx1[i, :] = np.power(X, (n+i+1)/n)
        Fx3[i, :] = np.power(X, n/(n+i+1))
        Fx2[i, :] = np.power(X, i+2)
        Fx4[i, :] = np.power(X, 1/(i+2))
    Y = np.vstack((X, Fx1))    
    Y = np.vstack((Y, Fx3))
    Y = np.vstack((Y, Fx2))
    Y = np.vstack((Y, Fx4))    
    return Y

def reconstruct(Y, Totalnum, nth):
#    if nth == 0:
#        X = Y
#    elif nth == Totalnum - 1:
#        X = 1 - Y
#    elif nth<=((Totalnum - 2)/4):
#        X = np.power(Y, 1/(nth + 1))
#    elif nth < (Totalnum/2):
#        X = np.power(1-Y, 1/(nth - (Totalnum - 2)/4 + 1))
#    elif nth < (Totalnum*3/4):
#        X = np.power(Y, nth - (Totalnum - 2)/2 + 1)
#    else:
#        X = np.power(1-Y, nth - (Totalnum - 2)*3/4 + 1)
    if nth == 0:
        X = Y
    elif nth<=((Totalnum - 1)/4):
        X = np.power(Y, ((Totalnum - 1)/4) /(((Totalnum - 1)/4) + nth))
    elif nth<=((Totalnum - 1)/2):
        X = np.power(Y, (nth/((Totalnum - 1)/4)))
    elif nth<=(3*(Totalnum - 1)/4):
        X = np.power(Y, 1/(nth - ((Totalnum - 1)/2) + 1))
    else:
        X = np.power(Y, nth - (3*(Totalnum - 1)/4) + 1)
    return X
       
def HFCM_ridge(dataset1, ratio=0.7, plot_flag=False):

    normalize_style = '01'
    dataset_copy = dataset1.copy()
    dataset, maxV, minV = normalize(dataset1, normalize_style)
    # dataset = 1 - dataset
    # dataset = dataset1
    # steepness of sigmoid function
    # belta = 1
    belta = 5   

    # partition dataset into train set and test set\
    if len(dataset) > 30:
        # ratio = 0.83
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data, test_data = splitData(dataset, 1)
        test_data = train_data

    len_train_data = len(train_data)
    len_test_data = len(test_data)
    # grid search
    # best parameters
    validation_ratio = 0.2
    len_validation_data = int(len_train_data * validation_ratio)

    # small_alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20]
    # small_alpha = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    # small_alpha = [ 1e-5,3e-5,5e-5,7e-5, 9e-5, 1e-4, 3e-4,5e-4,7e-4,9e-4]
    # small_alpha = [ 1e-7,3e-7,5e-7,7e-7, 9e-7, 1e-6, 3e-6,5e-6,7e-6,9e-6]
    # small_alpha = [1e-6, 5e-6, 1e-7,5e-7, 1e-8, 5e-8, 1e-9, 5e-9,  1e-10, 5e-10]
    # small_alpha = [1e-11, 5e-11, 1e-12,5e-12, 1e-13, 5e-13, 1e-14, 5e-14,  1e-15, 5e-15]
    # small_alpha = [1e-16, 5e-16, 1e-17,5e-17, 1e-18, 5e-18, 1e-19, 5e-19,  1e-20, 5e-20]
    # small_alpha = [1e-12, 1e-14, 1e-20]
    small_alpha = [1e-12]
    # small_alpha = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    # small_alpha = [1e-3, 1e-5, 1e-7, 1e-12, 1e-14, 1e-20]
    
    Order_list = list(range(2, 3))
    Nc_list = list(range(6, 7))
    alpha_list = small_alpha
    # rmse_total = np.zeros(shape=(len(Nc_list), len(Order_list))) 
    best_Order = -1
    best_Nc = -1
    # Nc = 6  # Fcm节点数
    Nf = 5 # 函数家族数除以4的值减1
    epochs = 50
    # 最优参数
    min_rmse = np.inf
    best_predictedTimeseries = np.zeros(shape = len_train_data +len_test_data)
    best_W_learned = None
    best_steepness = None
    best_alpha = -1
    
    for Nidx, Nc in enumerate(Nc_list):
        for snap in range(epochs):    
            Allinall = random_deposition(dataset, Nf)
            len_All, wth_All = np.shape(Allinall)
            location = random.sample(range(0, len_All), Nc)
            coffis = np.zeros((Nc, wth_All))
            for i in range(Nc):
                coffis[i, :] = Allinall[location[i], :]
            np.savetxt('coffis.txt', coffis, delimiter=',')
            for Oidx, Order in enumerate(Order_list):
                # Grid Search for optimizing alpha
                for alpha in alpha_list:                    
                    U_train = coffis[:, :len_train_data - len_validation_data]
        
                    # the ridge regression
                    tol = 1e-24
                    from sklearn import linear_model
                    # clf = linear_model.ElasticNet(alpha = alpha, l1_ratio = gamma, tol = tol)
                
                    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol)
                    # solving Ax = b to obtain x(x is the weight vector corresponding to certain node)
        
                    # learned weight matrix
                    W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
                    samples_train = {}
                    for node_solved in range(Nc):  # solve each node in turn
                        samples = create_dataset(U_train, belta, Order, node_solved)
                        # delete last "Order" rows (all zeros)
                        samples_train[node_solved] = samples[:-Order, :]
                        # use ridge regression
                        clf.fit(samples[:, :-1], samples[:, -1])
                        W_learned[node_solved, :] = clf.coef_
						
                    # # RCGA    
                    # W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
                    # samples_train = {}
                    # random.seed(time.time())
                    # min_fitness = 1e-3
                    # max_evals = 1000
                    # pop_size = 100
                    # num_variables = Nc * Order + 1
                    # xover_rate = 0.8
                    # mu_rate = 0.1
                    # max_limits = np.repeat(5, num_variables)
                    # min_limits = np.repeat(-5, num_variables)
                    # prob = 1
                    # num_experiments = 1
                    # num_xover = 0
                    # for node_solved in range(Nc):
                    #     samples = create_dataset(U_train, belta, Order, node_solved)
                    #     samples_train[node_solved] = samples[:-Order, :]
                    #     global_fit = np.zeros(num_experiments)
                    #     choose_xover  = num_xover + 1
                    #     for j in range(num_experiments):
                    #         evo = Evolution(prob, pop_size, num_variables, max_evals,  max_limits, min_limits, xover_rate, mu_rate, min_fitness, choose_xover, samples)
                    #         best, best_fit, global_best, global_bestfit, pop, fit_list = evo.evo_alg()
                    #         W_learned[node_solved, :] = global_best
                                                        
                    steepness = np.max(np.abs(W_learned), axis=1)
                    for i in range(Nc):
                        if steepness[i] > 1:
                            W_learned[i, :] /= steepness[i]
                    
                    # predict on training data set
                    trainPredict = np.zeros(shape=(Nc, len_train_data - len_validation_data))
                    for i in range(Nc):
                        trainPredict[i, :Order] = U_train[i, :Order] 
                        trainPredict[i, Order:] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)
                    trainPredict = revise(trainPredict)
        
                    # # reconstruct part
                    new_trainPredict = np.zeros(shape=(Nc, len_train_data - len_validation_data))
                    for i in range(Nc):
                        new_trainPredict[i, :] = reconstruct(trainPredict[i, :], 4 * Nf + 1, location[i])
                    sin_dimension_Train = np.sum(new_trainPredict, 0)/Nc
                    
                
                    # validation stage for choosing right parameters
                    U_validation = coffis[:, len_train_data - len_validation_data - Order:len_train_data]
                    validationPredict = np.zeros(shape=(Nc, len_validation_data))
                    samples_validation = {}
                    for i in range(Nc):  # solve each node in turn
                        samples2 = create_dataset(U_validation, belta, Order, i)
                        samples_validation[i] = samples2[:-Order, :]  # delete the last "Order' rows(all zeros)
                        # testPredict[i, :Order] = U_test[i, :Order]
                        validationPredict[i, :] = predict(samples_validation[i], W_learned[i, :], steepness[i], belta)
                    validationPredict = revise(validationPredict)
                    new_validationPredict = np.zeros(shape=(Nc, len_validation_data))
                    for i in range(Nc):
                        new_validationPredict[i, :] = reconstruct(validationPredict[i, :], 4 * Nf + 1, location[i])
                    sin_dimension_Validition = np.sum(new_validationPredict, 0)/Nc
                    
                    # # test data
                    U_test = coffis[:, len_train_data - Order:]   # use last Order data point of train dataset
                    testPredict = np.zeros(shape=(Nc, len_test_data))
                    samples_test = {}
                    for i in range(Nc):  # solve each node in turn
                        samples3 = create_dataset(U_test, belta, Order, i)
                        samples_test[i] = samples3[:-Order, :]  # delete the last "Order' rows(all zeros)
                        testPredict[i, :] = predict(samples_test[i], W_learned[i, :], steepness[i], belta)
                    testPredict = revise(testPredict)    
                    new_testPredict = np.zeros(shape=(Nc, len_test_data))
                    for i in range(Nc):
                        new_testPredict[i, :] = reconstruct(testPredict[i, :], 4 * Nf + 1, location[i])
                    sin_dimension_Test = np.sum(new_testPredict, 0)/Nc
                    
                     # print('Error is %f' % np.linalg.norm(np.array(train_data)[k:] - new_trainPredict, 2))
                    if plot_flag:
                        # plot train data series and predicted train data series
                        fig2 = plt.figure()
                        ax_2 = fig2.add_subplot(111)
                        ax_2.plot(dataset, 'r:', label='the original data')
                        tempva = np.hstack((sin_dimension_Train, sin_dimension_Validition))
                        tempva = np.hstack((tempva, sin_dimension_Test))
                        ax_2.plot(tempva, 'g:', label='the predicted data')
                        ax_2.set_xlabel('Year')
                        ax_2.set_title('time series(train dataset) by wavelet')
                        ax_2.legend()
                    
                    mse, rmse, nmse = statistics(dataset[len_train_data - len_validation_data:len_train_data], sin_dimension_Validition)
                    # rmse_total[Nidx, Oidx] = rmse
                    print("Nc -> %d, Order -> %d, alpha -> %g: rmse -> %f  |)"% (Nc, Order, alpha, rmse))
                    tempva = np.hstack((sin_dimension_Train, sin_dimension_Validition))
                    tempva = np.hstack((tempva, sin_dimension_Test))
                    # use rmse as performance index
                    if rmse < min_rmse:
                        min_rmse = rmse
                        best_Order = Order
                        best_Nc = Nc
                        best_predictedTimeseries = tempva
                        # best_predict[:] = np.hstack((sin_dimension_Train, sin_dimension_Validition))
                        best_W_learned = W_learned
                        best_steepness = steepness
                        best_alpha = alpha     
    # print(rmse_total)
    # best_predictedTimeseries = 1 - best_predictedTimeseries
    data_predicted = re_normalize(best_predictedTimeseries, maxV, minV, normalize_style)
    return data_predicted, best_Order, best_Nc, best_alpha


def analyze_paras_HFCM(dataset1, ratio=0.7):
    normalize_style = '01'
    dataset_copy = dataset1.copy()
    dataset, maxV, minV = normalize(dataset1, normalize_style)
    belta = 1

    # partition dataset into train set and test set\
    if len(dataset) > 30:
        # ratio = 0.83
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data, test_data = splitData(dataset, 1)
        test_data = train_data

    len_train_data = len(train_data)
    len_test_data = len(test_data)
    # grid search
    # best parameters
    validation_ratio = 0.2
    len_validation_data = int(len_train_data * validation_ratio)

    # small_alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20]
    # small_alpha = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    # small_alpha = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    # small_alpha = [1e-6, 5e-6, 1e-7,5e-7, 1e-8, 5e-8, 1e-9, 5e-9,  1e-10, 5e-10]
    # small_alpha = [1e-11, 5e-11, 1e-12,5e-12, 1e-13, 5e-13, 1e-14, 5e-14,  1e-15, 5e-15]
    # small_alpha = [1e-16, 5e-16, 1e-17,5e-17, 1e-18, 5e-18, 1e-19, 5e-19,  1e-20, 5e-20]
    # small_alpha = [ 1e-5,3e-5,5e-5,7e-5, 9e-5, 1e-4, 3e-4,5e-4,7e-4,9e-4]
    # small_alpha = [ 1e-7,3e-7,5e-7,7e-7, 9e-7, 1e-6, 3e-6,5e-6,7e-6,9e-6]
    small_alpha = [1e-3, 1e-12, 1e-14]
    
#    Order_list = list(range(2, 12))
#    Nc_list = list(range(2, 12))
    # for analysizing alpha
    Order_list = list(range(2, 4))
    Nc_list = list(range(6, 7))
    alpha_list = list(small_alpha)
    rmse_total = np.zeros(shape=(len(Nc_list), len(Order_list)))
    rmse_byalpha = np.zeros(len(alpha_list))
    # Nc = 6  # Fcm节点数
    Nf = 5 # 函数家族数除以4的值减1
    epochs = 20
    # 最优参数
    min_rmse = np.inf
    best_steepness = None

    for Nidx, Nc in enumerate(Nc_list):
        for snap in range(epochs):
            Allinall = random_deposition(dataset, Nf)
            len_All, wth_All = np.shape(Allinall)
            location = random.sample(range(0, len_All), Nc)
            coffis = np.zeros((Nc, wth_All))
            for i in range(Nc):
                coffis[i, :] = Allinall[location[i], :]
            np.savetxt('coffis.txt', coffis, delimiter=',')
            for Oidx, Order in enumerate(Order_list):
                # Grid Search for optimizing alpha
                for Aidx, alpha in enumerate(alpha_list):
                    U_train = coffis[:, :len_train_data - len_validation_data]
        
                    # the ridge regression
                    tol = 1e-24
                    from sklearn import linear_model
                    # clf = linear_model.ElasticNet(alpha = alpha, l1_ratio = gamma, tol = tol)
                
                    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol)
                    # solving Ax = b to obtain x(x is the weight vector corresponding to certain node)
        
                    # learned weight matrix
                    W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
                    samples_train = {}
                    for node_solved in range(Nc):  # solve each node in turn
                        samples = create_dataset(U_train, belta, Order, node_solved)
                        # delete last "Order" rows (all zeros)
                        samples_train[node_solved] = samples[:-Order, :]
                        # use ridge regression
                        clf.fit(samples[:, :-1], samples[:, -1])
                        W_learned[node_solved, :] = clf.coef_
        
                    steepness = np.max(np.abs(W_learned), axis=1)
                    for i in range(Nc):
                        if steepness[i] > 1:
                            W_learned[i, :] /= steepness[i]
                    
                    # predict on training data set
                    trainPredict = np.zeros(shape=(Nc, len_train_data - len_validation_data))
                    for i in range(Nc):
                        trainPredict[i, :Order] = U_train[i, :Order] 
                        trainPredict[i, Order:] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)
                    trainPredict = revise(trainPredict)
        
                    # # reconstruct part
                    new_trainPredict = np.zeros(shape=(Nc, len_train_data - len_validation_data))
                    for i in range(Nc):
                        new_trainPredict[i, :] = reconstruct(trainPredict[i, :], 4 * Nf + 1, location[i])
                    sin_dimension_Train = np.sum(new_trainPredict, 0)/Nc
                    
                
                    # validation stage for choosing right parameters
                    U_validation = coffis[:, len_train_data - len_validation_data - Order:len_train_data]
                    validationPredict = np.zeros(shape=(Nc, len_validation_data))
                    samples_validation = {}
                    for i in range(Nc):  # solve each node in turn
                        samples2 = create_dataset(U_validation, belta, Order, i)
                        samples_validation[i] = samples2[:-Order, :]  # delete the last "Order' rows(all zeros)
                        # testPredict[i, :Order] = U_test[i, :Order]
                        validationPredict[i, :] = predict(samples_validation[i], W_learned[i, :], steepness[i], belta)
                    validationPredict = revise(validationPredict)
                    new_validationPredict = np.zeros(shape=(Nc, len_validation_data))
                    for i in range(Nc):
                        new_validationPredict[i, :] = reconstruct(validationPredict[i, :], 4 * Nf + 1, location[i])
                    sin_dimension_Validition = np.sum(new_validationPredict, 0)/Nc
                    
                    # # test data
                    U_test = coffis[:, len_train_data - Order:]   # use last Order data point of train dataset
                    testPredict = np.zeros(shape=(Nc, len_test_data))
                    samples_test = {}
                    for i in range(Nc):  # solve each node in turn
                        samples3 = create_dataset(U_test, belta, Order, i)
                        samples_test[i] = samples3[:-Order, :]  # delete the last "Order' rows(all zeros)
                        testPredict[i, :] = predict(samples_test[i], W_learned[i, :], steepness[i], belta)
                    testPredict = revise(testPredict)    
                    new_testPredict = np.zeros(shape=(Nc, len_test_data))
                    for i in range(Nc):
                        new_testPredict[i, :] = reconstruct(testPredict[i, :], 4 * Nf + 1, location[i])
                    sin_dimension_Test = np.sum(new_testPredict, 0)/Nc
                                        
                    mse, rmse, nmse = statistics(dataset[len_train_data:], sin_dimension_Test)
                    # rmse_total[Nidx, Oidx] = rmse
                    print("Nc -> %d, Order -> %d, alpha -> %g: rmse -> %f  |)"% (Nc, Order, alpha, rmse))
                   
                    # use rmse as performance index
                    if rmse < min_rmse:
                        min_rmse = rmse                        
                        best_steepness = steepness
                    rmse_byalpha[Aidx] = rmse_byalpha[Aidx] + rmse
                rmse_total[Nidx, Oidx] = rmse_total[Nidx, Oidx] + min_rmse
    # 每个（Nc, Order）下最优alpha 时的误差(Validation dataset)
    rmse_byalpha = rmse_byalpha/epochs
    rmse_total = rmse_total/epochs
    df = pd.DataFrame(rmse_total, index=Nc_list, columns=Order_list)
    return df, rmse_byalpha


# analyze hyper-parameters on the performance on Wavelet-HFCM
def analyze_parameter():
    import seaborn as sns
    # Analyze sunspot and s&p 500 time series

    # data set : sunspot   
    ratio = 0.7
    
    sp500_src = "./newdata/^N225.csv" 
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    
    df1 ,rmse_byalpha1 = analyze_paras_HFCM(dataset, ratio=ratio)

    Nc_list = df1.index.values
    Order_list = df1.columns.values
    # Alpha_list = [1e-3, 1e-5, 1e-7, 1e-12, 1e-14, 1e-20]
    Alpha_list = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]

    
    sp500_src = "./newdata/^DJI (2).csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    df2 ,rmse_byalpha2= analyze_paras_HFCM(dataset, ratio=ratio)
    
    
    sp500_src = "./newdata/^TNX.csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    df3 ,rmse_byalpha3 = analyze_paras_HFCM(dataset, ratio=ratio)
    
    
    sp500_src = "./newdata/^IXIC (2).csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    df4  ,rmse_byalpha4= analyze_paras_HFCM(dataset, ratio=ratio)
    
    
    sp500_src = "./newdata/^GSPC (2).csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    df5  ,rmse_byalpha5= analyze_paras_HFCM(dataset, ratio=ratio)
    
    
    sp500_src = "./datasets/sp500.csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    df6  ,rmse_byalpha6= analyze_paras_HFCM(dataset, ratio=ratio)
    
    
    sp500_src = "./newdata/^RUT (2).csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    df7  ,rmse_byalpha7= analyze_paras_HFCM(dataset, ratio=ratio)
    
#    rmsealpha = np.vstack((rmse_byalpha1,rmse_byalpha2,rmse_byalpha3,rmse_byalpha4,rmse_byalpha5,rmse_byalpha6,rmse_byalpha7))
#    dfalpha = pd.DataFrame(rmsealpha, index=list(range(1, 8)), columns=Alpha_list)
    # save df1 & df2 to excel
    writer = pd.ExcelWriter('output_sunspot_sp500.xlsx')
    df1.to_excel(writer, 'df1')
    df2.to_excel(writer, 'df2')
    df3.to_excel(writer, 'df3')
    df4.to_excel(writer, 'df4')
    df5.to_excel(writer, 'df5')
    df6.to_excel(writer, 'df6')
    df7.to_excel(writer, 'df7')
    writer.save()

    
    # RMSE versus varying level of decomposition
    # sunspot + S&P 500
    import shutil
    import os
    plt.style.use(['seaborn-paper'])
    # sns.set_style("dark")
    if not os.path.exists('./Outcome_for_papers/impact_parameters/varying_Nc'):
        os.makedirs('./Outcome_for_papers/impact_parameters/varying_Nc')
    if not os.path.exists('./Outcome_for_papers/impact_parameters/varying_Order'):
        os.makedirs('./Outcome_for_papers/impact_parameters/varying_Order')
    if not os.path.exists('./Outcome_for_papers/impact_parameters/varying_Alpha'):
        os.makedirs('./Outcome_for_papers/impact_parameters/varying_Alpha')
        
    df = pd.DataFrame({r'$alpha$': Alpha_list, 'RUT': rmse_byalpha7,'SP500':rmse_byalpha6,'GSPC': rmse_byalpha5,
                       'IXIC': rmse_byalpha4, 'TNX': rmse_byalpha3,'DJI': rmse_byalpha2,'N225': rmse_byalpha1})
    
    df = pd.melt(df, id_vars=r'$alpha$', var_name="Dataset", value_name='RMSE')
    g = sns.factorplot(x=r'$alpha$', y='RMSE', hue='Dataset',
                       hue_order=['RUT','SP500','GSPC','IXIC', 'TNX','DJI', 'N225'], data=df, kind='bar',
                       legend=True, palette=sns.color_palette(["#95a5a6", "#34495e", "#9b59b6", "#3498db",  "#e74c3c", "#95a5a6", "#2ecc71"]))

    # resize figure box to -> put the legend out of the figure
    box = g.ax.get_position()  # get position of figure
    g.ax.set_position([box.x0, box.y0, box.width, box.height])  # resize position

    # Put a legend to the right side
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    # plt.tight_layout()

    plt.savefig(
        r"./Outcome_for_papers/impact_parameters/varying_alpha/alpha.pdf")
    plt.savefig(
        r"./Outcome_for_papers/impact_parameters/varying_alpha/alpha.tiff")
    plt.savefig(
        r"./Outcome_for_papers/impact_parameters/varying_alpha/alpha.png")
    plt.close()
    
        
#    for order in Order_list:
#        df = pd.DataFrame({r'$N_c$': Nc_list, 'RUT': df7[order].values, 'SP500': df6[order].values,'GSPC': df5[order].values,
#                           'IXIC': df4[order].values, 'TNX': df3[order].values, 'DJI': df2[order].values, 'N225': df1[order].values})
#        df = pd.melt(df, id_vars=r'$N_c$', var_name="Dataset", value_name='RMSE')
#        g = sns.factorplot(x=r'$N_c$', y='RMSE', hue='Dataset',
#                           hue_order=['N225', 'DJI', 'TNX', 'IXIC', 'GSPC', 'SP500', 'RUT'], data=df, kind='bar',
#                           legend=True, palette=sns.color_palette(["#34495e", "#95a5a6", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#2ecc71"]))
#
#        # resize figure box to -> put the legend out of the figure
#        box = g.ax.get_position()  # get position of figure
#        g.ax.set_position([box.x0, box.y0, box.width, box.height])  # resize position

#        # Put a legend to the right side
#        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
#        # plt.tight_layout()
#
#        plt.savefig(
#            r"./Outcome_for_papers/impact_parameters/varying_Nc/k=%d.pdf" % order)
#        plt.savefig(
#            r"./Outcome_for_papers/impact_parameters/varying_Nc/k=%d.tiff" % order)
#        plt.savefig(
#            r"./Outcome_for_papers/impact_parameters/varying_Nc/k=%d.png" % order)
#        plt.close()
#
#    for Nc in Nc_list:
#        # / print(len(df_1.loc[Nc, :]))
#        df = pd.DataFrame({'$k$': Order_list, 'RUT': df7.loc[Nc, :].values, 'SP500': df6.loc[Nc, :].values,'GSPC': df5.loc[Nc, :].values,
#                           'IXIC': df4.loc[Nc, :].values, 'TNX': df3.loc[Nc, :].values, 'DJI': df2.loc[Nc, :].values, 'N225': df1.loc[Nc, :].values})
#
#        df = pd.melt(df, id_vars='$k$', var_name="Dataset", value_name='RMSE')
#        g = sns.factorplot(x='$k$', y='RMSE', hue='Dataset',
#                           hue_order=['N225', 'DJI', 'TNX', 'IXIC', 'GSPC', 'SP500', 'RUT'], data=df, kind='bar',
#                           legend=True, palette=sns.color_palette(["#34495e", "#95a5a6", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#2ecc71"]))
#
#        # resize figure box to -> put the legend out of the figure
#        box = g.ax.get_position()  # get position of figure
#        g.ax.set_position([box.x0, box.y0, box.width, box.height])  # resize position
#
#        # Put a legend to the right side
#        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
#
#        plt.savefig(
#            r"./Outcome_for_papers/impact_parameters/varying_Order/Nc=%d.pdf" % Nc)
#        plt.savefig(
#            r"./Outcome_for_papers/impact_parameters/varying_Order/Nc=%d.tiff" % Nc)
#        plt.savefig(
#            r"./Outcome_for_papers/impact_parameters/varying_Order/Nc=%d.png" % Nc)
#        plt.close()
#



def main():
    # load time series data

    ''' New data sets'''
    # data set 3 : sp500 index
    sp500_src = "./datasets/sp500.csv"
    # sp500_src = "./data_inpaper/^DJI (2).csv"
    # sp500_src = "./data_inpaper/^GSPC (2).csv"
    # sp500_src = "./data_inpaper/^IXIC (2).csv"
    # sp500_src = "./data_inpaper/^RUT (2).csv" #可用于网格搜索的超参数验证的序列
    # sp500_src = "./data_inpaper/^N225.csv" #可用于网格搜索的超参数验证的序列
    # sp500_src = "./data_inpaper/^TNX.csv" 
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse)
    dataset = np.array(sp500.iloc[:, 1], dtype=np.float)
    time = np.array(sp500.iloc[:, 0])
    ratio = 0.7
#    raw_data = pd.read_csv("./newdata/nasdaq100_padding.csv")
#    dataset = raw_data['NDX']
#    tempp = len(dataset)
#    time = np.zeros(tempp)
#    for i in range(tempp):
#        time[i] = i
#    ratio = 0.7 
    
    # partition dataset into train set and test set
    length = len(dataset)
    len_train_data = int(length * ratio)

    validation_ratio = 0.2
    len_validation_data = int(len_train_data * validation_ratio)
    len_test_data = length - len_train_data

    # perform prediction
    data_predicted, best_Order, best_Nc, best_alpha = HFCM_ridge(dataset, ratio)
    import seaborn as sns
    '''
    # first line for plot in paper 
    normalize_style = '01'
    dataset_copy = dataset.copy()
    datasetplot, maxV, minV = normalize(dataset_copy, normalize_style)
    Allinall = random_deposition(datasetplot, 5)
    len_All, wth_All = np.shape(Allinall)
    location = random.sample(range(0, len_All), 8)
    coffis = np.zeros((8, wth_All))
    pre = np.zeros(shape = [8, wth_All])
    for i in range(8):
        coffis[i, :] = Allinall[location[i], :]
    for i in range(8):
        pre[i, :] = re_normalize(coffis[i, :], maxV, minV, normalize_style)
        plt.style.use(['ggplot','seaborn-paper'])
        fig4 = plt.figure()
        ax41 = fig4.add_subplot(111)
        ax41.plot(time, pre[i, :], 'm:')
        plt.savefig(r"C:/Users/Administrator/Desktop/idea190910/sp500fearturetime = %d.png" % i)
#        ax41.plot(time, pre[i, :], 'b:')
#        plt.savefig(r"C:/Users/Administrator/Desktop/idea190910/predictedtime = %d.png" % i)
        plt.show()
        
    # last line for plot in paper 
    '''
    
    # Outcomes
    # Error of the whole dataset
    mse, rmse, nmse = statistics(dataset, data_predicted)
    print('*' * 80)
    print('The ratio is %f' % ratio)
    print('best Order is %d, best Nc is %d, best alpha is %g' % (best_Order, best_Nc, best_alpha))
    print('Forecasting on all dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))

    # Error of Train dataset
    mse, rmse, nmse = statistics(dataset[:len_train_data-len_validation_data], data_predicted[:len_train_data-len_validation_data])
    print('Forecasting on train dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))

    # Error of Validation dataset
    mse, rmse, nmse = statistics(dataset[len_train_data-len_validation_data:len_train_data], data_predicted[len_train_data-len_validation_data:len_train_data])
    print('Forecasting on validation dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))

    # Error of Test dataset
    mse, Test_rmse, nmse = statistics(dataset[len_train_data:], data_predicted[len_train_data:])
    print('Forecasting on test dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(Test_rmse, 2), Test_rmse, nmse))

    # print length of each subdatasets

    print('The whole length is %d' % length)
    print('Train dataset length is %d' % (len_train_data - len_validation_data))
    print('Validation dataset length is %d' % len_validation_data)
    print('Test dataset length is %d' % len_test_data)


    # plot time series
    import seaborn as sns
    # plt.style.use(['seaborn-paper'])
    plt.style.use(['ggplot','seaborn-paper'])

    fig4 = plt.figure()
    ax41 = fig4.add_subplot(111)
    

    ax41.plot(time, dataset, 'r:', label='the original data')
    ax41.plot(time, data_predicted, 'k:', label='the predicted data')
    # ax41.plot(time[len_train_data:], data_predicted[len_train_data:], 'b:', label='the predicted data')
    ax41.set_ylabel("Magnitude")
    ax41.set_xlabel('Time')
    # ax41.set_title('time series prediction ')

    # ax41.set_ylim([0.35, 1.4])  # for MG-chaos having a better visualization

    ax41.legend()
    plt.tight_layout()
   
    # ['N225', 'DJI', 'TNX', 'IXIC', 'GSPC', 'SP500', 'RUT']
    # plt.savefig(r"./Outcome_for_papers/Length_TNX=%d.tiff" % length)
    # plt.savefig(r"./Outcome_for_papers/Length=%d.pdf" % length)
    if not os.path.exists(r"./Outcome_for_papers"):
        os.makedirs(r"./Outcome_for_papers")
    plt.savefig(r"./Outcome_for_papers/Length_TNX1008 = %d.png" % length)
    # plt.close()
    plt.show()

def statistics(origin, predicted):
    # # compute RMSE
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(origin, predicted)
    rmse = np.sqrt(mse)
    meanV = np.mean(origin)
    dominator = np.linalg.norm(predicted - meanV, 2)
    return mse, rmse, mse / np.power(dominator, 2)


if __name__ == '__main__':
    # analyze hyper-parameters on the performance of Wavelet-HFCM
    # analyze_parameter()
    # main function
    main()

