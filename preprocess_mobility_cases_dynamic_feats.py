from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
import os


def prep_data(data, nos_of_days, covariates, counties, stat_feats, data_start, train=True):
    
    print ("CHECK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print ("DATA:", np.shape(data))
    print ("NOS OF DAYS:", nos_of_days)
    print ("COVARIATES:", np.shape(covariates))
    print ("counties:",counties)
    print ("stat_feats:",stat_feats)
    print ("data_start:",np.shape( data_start))
    print ("CHECK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    
    
    
    # print("train: ", train)
    time_len = data.shape[0]  # total length of time series #32136
    #print("time_len: ", time_len) #
    input_size = window_size - stride_size  # 192 - 24
    #print ("INPUT SIZE:", input_size)
    # num_series = 50
    # print("NUM SERIES:", num_series)
    # print ("Data start:", np.shape(data_start))
    windows_per_series = np.full((num_series),
                                 (time_len - input_size) // stride_size)  # create 370(num_series)*1 dimensional vector
    # print("windows_per_series pre: ", windows_per_series.shape)
    if train: windows_per_series -= (
                                                data_start + stride_size - 1) // stride_size  # hop through the entire time series with windows rain/test
    # print("data_start: ", np.shape(data_start)) #(370,)
    # print ("Type data start:", type(data_start))
    # print("data_start:",data_start) #(370,)
    # print("windows_per_series: ", windows_per_series.shape) #(370,)
    # print("windows_per_series:", windows_per_series)
    # print ("Type of windows_per_series:", type(windows_per_series)) #<class 'numpy.ndarray'>

    total_windows = np.sum(windows_per_series)
    # print ("total_windows:", total_windows) #total_windows #6490

    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    #print ("X INPUT SHAPE:", x_input.shape)
    # x_input = np.zeros((total_windows, window_size, 50, 1 + num_covariates + 1), dtype='float32')

    # print ("Shape of x_input:", np.shape(x_input))
    label = np.zeros((total_windows, window_size), dtype='float32')
    # print("Shape of label:", np.shape(label))
    v_input = np.zeros((total_windows, 2), dtype='float32')

    ## for dynamic 2 node features # reshape covariate dimensions
    covariates = covariates.reshape(int(nos_of_days),counties * stat_feats)
    print("Shape of covariates after reshaping:", np.shape(covariates))

    # cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    # cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
    count = 0
    if not train:
        covariates = covariates[-time_len:]
        # print ("shape of covariates not train:",(covariates))

    # print("num_series:", num_series) #3245
    for series in trange(num_series):  # 370
        # print ("series:", series)
        cov_age = stats.zscore(np.arange(total_time - data_start[series]))
        # print ("cov_age:", np.shape(cov_age)) # 150*1
        if train:
            # print ("Data start:", data_start[series])
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len - data_start[series]]
            # covariates[data_start[series]:time_len, 0,0] = cov_age[:time_len - data_start[series]]
            # print("covariates in train:", np.shape(covariates))
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size * i + data_start[series]
                # print ("window_start:", window_start)
            else:
                window_start = stride_size * i
            window_end = window_start + window_size

            # print("x: ", x_input[count, 1:, 0].shape)
            # print("window start: ", window_start)
            # print("window end: ", window_end)
            # print("data: ", data.shape)
            # print("d: ", data[window_start:window_end-1, series].shape)

            # print("X-input shape:", np.shape(x_input))
            # print("covariates shape:", np.shape(covariates))

            # x_input[count, 1:, 0] = data[window_start:window_end - 1, series]
            # x_input[count, :, 1:1 + num_covariates] = covariates[window_start:window_end, :]
            
            #print ("COUNT:", count)
            # print ("WINDOW START:",window_start )
            # print ("WINDOW END:",window_end )
            # print ("SERIES:", series)
            x_input[count, 1:, 0] = data[window_start:window_end - 1, series]
            x_input[count, :, 1:1 + num_covariates] = covariates[window_start:window_end, :]
            #
            # print ("x_input shape:", np.shape(x_input))

            # train_data = data_frame[1:140].values

            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
            # nonzero_sum = (x_input[count, 1:input_size, 1:3245,0] != 0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(), nonzero_sum) + 1
                x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]
                # v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 1:3245,0].sum(), nonzero_sum) + 1
                # x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]

                if train:
                    label[count, :] = label[count, :] / v_input[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix + 'data_' + save_name, x_input)
    np.save(prefix + 'v_' + save_name, v_input)
    np.save(prefix + 'label_' + save_name, label)





### with dynamic time series node features
def gen_covariates(num_covariates, counties, stat_feats,csv_files,nos_of_days):
        covariates = np.zeros(((nos_of_days),counties,stat_feats) ) ####  nos of days * nodes(counties) * static features
        print ("COVARIATES SHAPE:", np.shape(covariates))
        #covariates = np.zeros(((nos_of_days), counties, counties))
        #print ("Shape of covariates init:", np.shape(covariates))
        i = 0 # keep track of nos of days

        print ("COUNTIES:", counties)
        print ("LEN OF CSV FILES:", len(csv_files))
        print ("ENTERING FOR LOOP")
        for day in csv_files[0:150]: ## nos of days of the covariate matrix ## change

            print ("DAYS:", day)
            gcn_data = pd.read_csv(day)
            #print ("GCN DATA:", gcn_data)

            #print ("GCN data:",gcn_data.iloc[0, 1])
           
            #print ("i is:", i)
            #data = gcn_data.iloc[0, 0]
            for j in range(counties):
                #print ("J is:", j)
                for k in range(0,stat_feats ): # columns of the covariate matrix Gc*X
                    #print ("k is:", k)
                    covariates[i,j, k] = gcn_data.iloc[j, k]
            i = i + 1
            # if i == nos_of_days -1:
            #     break
                #covariates[i, j, 5] = input_time.month

            for k in range(0,stat_feats):
                covariates[:,:, k] = stats.zscore(covariates[:, :, k])
        return covariates[:, :, :stat_feats]




def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start + window_size], color='b')
    f.savefig("visual.png")
    plt.close()


if __name__ == '__main__':

    global save_path
    # name = 'LD2011_2014.txt'
   #name = "county_cases_data.csv" #####################
    name = "COUNTY_DEATHS_DATA.csv"
    ### Add every day dynamic features
    path = "/content/gdrive/My Drive/covariate_mat_300_CASES_MOBILITY_ORIG_200V_V1/"
    csv_files = glob.glob(os.path.join(path, "*.csv"))  # nos of days * covariate feats # covariate matrix
    counties = 277
    stat_feats = 2 ###########################
    save_name = 'elect'
    window_size = 12
    stride_size = 4
    # for static node features
    num_covariates = counties * stat_feats ## dynamic feats time series
    #num_covariates = 2  ## simple time series

    train_start = '2020-12-29 00:00:00'
    train_end = '2020-12-22 00:00:00'
    test_start = '2020-12-12 00:00:00'  # need additional 7 days as given info
    test_end = '2020-12-30 00:00:00'

    pred_days = 7
    given_days = 7
    #nos_of_days = 135 #len (csv_files)
    nos_of_days = 150 # simple time series
    #nos_of_days = 99 #number of days of the predictor ## NY covid death counts.
    # print ("nos_of_days:", nos_of_days)
    # print ("Type:", type(nos_of_days))

    save_path = os.path.join('data', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_path = os.path.join(save_path, name)
    if not os.path.exists(csv_path):
        zipurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(save_path)

    # data_frame = pd.read_csv(csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
    colums = [col for col in range(counties)]  ## Take 300 counties #first column being date
    print ("CSV PATH:", csv_path)
    data_frame = pd.read_csv(csv_path)


    #print("Shape of data_frame:", np.shape(data_frame))

    # for dynamic 2 feats nodes
    #data_frame = data_frame.iloc[2:93]  # take 68 days from Oct 16
    data_frame = data_frame.iloc[0:150,:] #Z ########################### check
    #print (data_frame)
    

    # data_frame = data_frame.iloc[:,[range(49)]]
    data_frame = data_frame.drop(["Days"], axis=1)######################################### check
    #data_frame = data_frame.iloc[:,0:counties] ######################## check
    print ("data_frame:", data_frame)
    print("Shape of data_frame:", np.shape(data_frame))


    # print("Type of data_frame:", type(data_frame))

    # for dynamic feats time searies
    covariates = gen_covariates(num_covariates, counties,stat_feats, csv_files,nos_of_days)

    # For simple time series
    #covariates = gen_covariates(num_covariates)
    print("SHAPE OF COVARIATES:", np.shape(covariates))  # 150 * 50 * 5

    # data_frame = data_frame.resample('1H', label='left', closed='right').sum()[train_start:test_end]
    # data_frame.fillna(0, inplace=True)
    # covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)

    # print ("Shape of covariates:", np.shape(covariates)) #(32304, 4)

    # train_data = data_frame[train_start:train_end].values
    # test_data = data_frame[test_start:test_end].values

    # for 2 dynamic node time series
    #train_data = data_frame[0:130].values # take training data ~ condiitoning range
    #test_data = data_frame[121:137].values # test data ~ prediction range

    # train_data = data_frame[0:128].values # take training data ~ condiitoning range
    # test_data = data_frame[120:135].values # test data ~ prediction range

    # for simple time series
    train_data = data_frame[0:143].values  # take training data ~ condiitoning range
    test_data = data_frame[135:150].values  # test data ~ prediction range

    print("Shape of train data:", np.shape(train_data))  # (32136, 370)
    print("Shape of test data:", np.shape(test_data))  # (336, 370)

    data_start = (train_data != 0).argmax(axis=0)  # find first nonzero value in each time series
    # data_start = 50
    total_time = data_frame.shape[0]  # 32304
    num_series = data_frame.shape[1]  # 370
    prep_data(train_data, nos_of_days, covariates, counties, stat_feats, data_start)
    prep_data(test_data, nos_of_days, covariates, counties, stat_feats, data_start, train=False)
