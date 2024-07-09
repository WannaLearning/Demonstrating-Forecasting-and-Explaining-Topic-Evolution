#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:04:15 2023

@author: aixuexi
"""
import os 
import re
import json
import math
import time
import string
import pickle
import random
import sklearn
import multiprocessing
import pandas as pd
import seaborn as sns 
import prettytable as pt
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt 
from tqdm import tqdm
from matplotlib import rcParams
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.optimize import curve_fit
from sklearn import decomposition
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

from Extract_cs_data import *
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
# 对时间回归的linear model
def linear_curve_fit(Y, func_type, lr=1e-1, num_iters=1000):
    time_len, dim_len = Y.shape
    Y_Real = Y.T
    X  = np.arange(time_len)[np.newaxis, :] + 1
    X  = (X - 1) / (time_len - 1)
    
    if func_type == 'y=ax+b':
        A = np.random.normal(size=dim_len)[:, np.newaxis]
        B = np.random.normal(size=dim_len)[:, np.newaxis]    
        params_init = np.concatenate([A, B], axis=-1)
        
        def linear_func(params, m=0):
            A, B   = params[:, 0:1], params[:, 1:2]
            Y_pred = np.matmul(A, X) + B
            return Y_pred
        
    elif func_type == 'y=ax2+bx+c':
        A = np.random.normal(size=dim_len)[:, np.newaxis]
        B = np.random.normal(size=dim_len)[:, np.newaxis]
        C = np.random.normal(size=dim_len)[:, np.newaxis]
        params_init = np.concatenate([A, B, C], axis=-1)
        
        def linear_func(params, m=0):
            A, B, C = params[:, 0:1], params[:, 1:2], params[:, 2:3]
            Y_pred = np.matmul(A, X ** 2) + np.matmul(B, X) + C
            return Y_pred
          
    elif func_type == 'y=ax3+bx2+cx+d':
        A = np.random.normal(size=dim_len)[:, np.newaxis]
        B = np.random.normal(size=dim_len)[:, np.newaxis]
        C = np.random.normal(size=dim_len)[:, np.newaxis]
        D = np.random.normal(size=dim_len)[:, np.newaxis]
        params_init = np.concatenate([A, B, C, D], axis=-1)
        
        def linear_func(params, m=0):
            A, B, C, D = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
            Y_pred = np.matmul(A, X ** 3) + np.matmul(B, X ** 2) + np.matmul(C, X) + D
            # Reg = np.mean(A ** 2) + np.mean(B ** 2)
            return Y_pred

    def callback(params, m, g):
        if m % 10 == 0:
            # print("Iteration {} MSE value {:.8f}".format(m, compute_loss(params, m)))
            pass
    
    def compute_loss(params, m=0):
        Y_pred = linear_func(params)
        Y_MSE  = np.mean((Y_Real - Y_pred) ** 2)
        return Y_MSE

    def eval_model(params):
        Y_pred = linear_func(params)
        eval_train = RMSE_MAE_MAPE(Y_Real, Y_pred)
        return eval_train

    gradient  = grad(compute_loss)
    params_final = adam(gradient, params_init, step_size=lr, num_iters=num_iters, callback=callback)
    eval_train = eval_model(params_final)

    return params_final, eval_train


def train_lr(train_input, train_output, test_output, func_type):
    # 均值 & 标准差 分别用线性函数拟合
    lr_models = dict()
    i = 0
    for fos in tqdm(train_input):
        i += 1
        lr_models[fos] = dict()
        # 拟合数据
        mu_ts  = list()
        std_ts = list()
        for t in train_input[fos]:
            mu_t  = train_input[fos][t][0]
            std_t = train_input[fos][t][1]
            mu_ts.append(mu_t)
            std_ts.append(std_t)
        for t in train_output[fos]:
            mu_t = train_output[fos][t][0]
            std_t = train_output[fos][t][1]
            mu_ts.append(mu_t)
            std_ts.append(std_t)
        mu_ts  = np.array(mu_ts)
        std_ts = np.array(std_ts)
        
        # 线性函数拟合
        parmas_mu,  eval_train_mu  = linear_curve_fit(mu_ts,  func_type)
        parmas_std, eval_train_std = linear_curve_fit(std_ts, func_type)
        # 
        lr_models[fos]["mu_ts"]      = mu_ts
        lr_models[fos]["std_ts"]     = std_ts
        lr_models[fos]["parmas_mu"]  = parmas_mu
        lr_models[fos]["parmas_std"] = parmas_std
        lr_models[fos]["eval_train_mu"] = eval_train_mu
        lr_models[fos]["eval_train_std"] = eval_train_std

    return lr_models
    

def train_lr_MP(train_input, train_output, test_output, model_name, func_type, mp_num=8):
    # 均值 & 标准差分别用线性函数拟合
    # 多进程调用 train_lr 函数
    fos_list  = list(train_input.keys())
    total_num = len(fos_list)
    batch_num = math.ceil(total_num/ mp_num)
    start_idx = 0
    end_idx   = 0
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        end_idx        = min(start_idx + batch_num, total_num)
        fos_list_i     = fos_list[start_idx: end_idx]
        train_input_i  = {fos: train_input[fos]  for fos in fos_list_i} 
        train_output_i = {fos: train_output[fos] for fos in fos_list_i} 
        test_output_i  = {fos: test_output[fos]  for fos in fos_list_i} 
        start_idx      = end_idx
        results.append(pool.apply_async(train_lr, (train_input_i, train_output_i, test_output_i, func_type)))
    pool.close()
    pool.join()
    
    lr_models = dict()
    for res in results:
        lr_models_i = res.get()
        for fos in lr_models_i:
            lr_models[fos] = lr_models_i[fos]
    save_file(lr_models, "./Models/lr_models({})({}).pkl".format(model_name, func_type))    
    
    
def test_lr(test_output, model_name, func_type):
    # 检查train_lr_MP拟合的结果在预测区间的loss
    if func_type == 'y=ax+b':
        def linear_func(params, X):
            A, B   = params[:, 0:1], params[:, 1:2]
            Y_pred = np.matmul(A, X) + B
            return Y_pred
    elif func_type == 'y=ax2+bx+c':
        def linear_func(params, X):
            A, B, C = params[:, 0:1], params[:, 1:2], params[:, 2:3]
            Y_pred = np.matmul(A, X ** 2) + np.matmul(B, X) + C
            return Y_pred
    elif func_type == 'y=ax3+bx2+cx+d':
        def linear_func(params, X):
            A, B, C, D = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
            Y_pred = np.matmul(A, X ** 3) + np.matmul(B, X ** 2) + np.matmul(C, X) + D
            return Y_pred
    

    def test_model(params, X_pred, Y_Real, t):
        Y_pred = linear_func(params, X_pred[:, :t])
        eval_train = RMSE_MAE_MAPE(Y_Real[:, :t], Y_pred)
        return eval_train
    
    lr_models = read_file("./Models/lr_models({})({}).pkl".format(model_name, func_type))
    lr_res    = dict()
    for fos in lr_models:
        if fos not in test_output:
            continue
        # 获取训练数据和训练结果 
        mu_ts          = lr_models[fos]["mu_ts"]
        std_ts         = lr_models[fos]["std_ts"] 
        parmas_mu      = lr_models[fos]["parmas_mu"]
        parmas_std     = lr_models[fos]["parmas_std"]
        eval_train_mu  = lr_models[fos]["eval_train_mu"]
        eval_train_std = lr_models[fos]["eval_train_std"]
        
        # 获取待预测的真实数据
        mu_ts_real  = list()
        std_ts_real = list()
        for t in test_output[fos]:
            mu_t  = test_output[fos][t][0]
            std_t = test_output[fos][t][1]
            mu_ts_real.append(mu_t)
            std_ts_real.append(std_t)
        mu_ts_real  = np.array(mu_ts_real).T
        std_ts_real = np.array(std_ts_real).T
        
        # 获取预测长度
        time_len_fit,  dim_len = mu_ts.shape
        time_len_real, dim_len = mu_ts_real.shape
        X_fit  = np.arange(time_len_fit)[np.newaxis, :] + 1
        X_pred = np.arange(time_len_fit, time_len_fit + time_len_real)[np.newaxis, :] + 1
        X_fit  = (X_fit  - 1) / (time_len_fit - 1)
        X_pred = (X_pred - 1) / (time_len_fit - 1)
        
        # mu & std prediction
        eval_test_mu_5   = test_model(parmas_mu, X_pred, mu_ts_real, 5)
        eval_test_mu_7   = test_model(parmas_mu, X_pred, mu_ts_real, 7)
        eval_test_mu_10  = test_model(parmas_mu, X_pred, mu_ts_real, 10)
        
        eval_test_std_5  = test_model(parmas_std, X_pred, std_ts_real, 5)
        eval_test_std_7  = test_model(parmas_std, X_pred, std_ts_real, 7)
        eval_test_std_10 = test_model(parmas_std, X_pred, std_ts_real, 10)
        
        lr_res[fos] = dict()
        lr_res[fos]['TEST_MU']  = np.array([list(eval_test_mu_5),  list(eval_test_mu_7),  list(eval_test_mu_10)])
        lr_res[fos]['TEST_STD'] = np.array([list(eval_test_std_5), list(eval_test_std_7), list(eval_test_std_10)])
        lr_res[fos]['TRAIN']    = np.array([list(eval_train_mu),   list(eval_train_std)])
        
    return lr_res


def test_lr_MP(train_input, train_output, test_output, model_name, func_type, mp_num=8):
    # 均值 & 标准差分别用线性函数拟合
    # 多进程调用 train_lr 函数
    fos_list  = list(train_input.keys())
    total_num = len(fos_list)
    batch_num = math.ceil(total_num/ mp_num)
    start_idx = 0
    end_idx   = 0
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        end_idx = min(start_idx + batch_num, total_num)
        fos_list_i = fos_list[start_idx: end_idx]
        test_output_i  = {fos: test_output[fos]  for fos in fos_list_i} 
        start_idx = end_idx
        results.append(pool.apply_async(test_lr, (test_output_i, model_name, func_type)))
    pool.close()
    pool.join()
    
    lr_res = dict()
    for res in results:
        lr_res_i = res.get()
        for fos in lr_res_i:
            lr_res[fos] = lr_res_i[fos]
               
    save_file(lr_res, "./Models/lr_res({})({}).pkl".format(model_name, func_type))
    

def lr_plot_case(train_input, train_output, test_output, model_name, fos=''):
    # 检查vector_auto_regression_MP的预测精度: 预测案例
    def linear_func(params, X):
        A, B   = params[:, 0:1], params[:, 1:2]
        Y_pred = np.matmul(A, X) + B
        return Y_pred
    
    lr_models = read_file("./Models/lr_models({})({}).pkl".format(model_name, 'y=ax+b'))
    
    if fos == "":
        fos = random.sample(train_input.keys(), 1)[0]
    # 拟合数据
    X_fit_mu  = list()
    X_fit_std = list()
    for t in train_input[fos]:
        X_fit_mu.append(train_input[fos][t][0])
        X_fit_std.append(train_input[fos][t][1])
    for t in train_output[fos]:
        X_fit_mu.append(train_output[fos][t][0])
        X_fit_std.append(train_output[fos][t][1])
    X_fit_mu  = np.array(X_fit_mu)
    X_fit_std = np.array(X_fit_std)
    # 预测数据
    Y_pred_mu  = list()
    Y_pred_std = list()
    for t in test_output[fos]:
        Y_pred_mu.append(test_output[fos][t][0])
        Y_pred_std.append(test_output[fos][t][1])
    Y_pred_mu  = np.array(Y_pred_mu)
    Y_pred_std = np.array(Y_pred_std)
    
    # 线性拟合参数
    parmas_mu  = lr_models[fos]["parmas_mu"]
    parmas_std = lr_models[fos]["parmas_std"]
        
    # 获取预测长度
    time_len_fit,  dim_len = X_fit_mu.shape
    time_len_real, dim_len = Y_pred_mu.shape
    X_fit  = np.arange(time_len_fit)[np.newaxis, :] + 1
    X_pred = np.arange(time_len_fit, time_len_fit + time_len_real)[np.newaxis, :] + 1
    X_fit  = (X_fit  - 1) / (time_len_fit - 1)
    X_pred = (X_pred - 1) / (time_len_fit - 1)

    # 模型拟合
    Y_hat_pred_mu = linear_func(parmas_mu,  X_pred).T
    
    # 模型评价
    _, bert_dim = X_fit_mu.shape
    r2_mu = list()
    for i in range(bert_dim):
        r2_mu_i = r2_score(Y_pred_mu[:, i], Y_hat_pred_mu[:, i])
        r2_mu.append(max(0, r2_mu_i))
    r2_mu = np.mean(r2_mu)
    dim_i = random.sample(list(np.arange(bert_dim)), 1)[0]
    r2_mu_i = r2_score(Y_pred_mu[:, dim_i], Y_hat_pred_mu[:, dim_i])
    
    # 绘图
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    ax = fig.add_subplot(111)

    Y_real = np.concatenate([X_fit_mu, Y_pred_mu], axis=0)
    Y_real = Y_real[:, dim_i]
    X_real = np.arange(len(Y_real))
    Y_pred = Y_hat_pred_mu[:, dim_i]
    X_pred = np.arange(len(Y_pred)) + len(X_fit_mu)
    
    ax.plot(X_real, Y_real, c='blue', label='Observed', marker='o', markersize=10)
    ax.plot(X_pred, Y_pred,  c='red',  label='Forecast', marker='s', linestyle='--', linewidth=3) # 'Forecast ({:.3f}, {:.3f})'.format(r2_mu_i, r2_mu)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_y   = math.ceil(max(max(Y_real),   max(Y_pred)) * 1000)
    down_y = math.floor( min(min(Y_real), min(Y_pred)) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    
    ax.fill_between(X_real[: len(X_fit_mu)], 
                    [yticks[0]] * len(X_fit_mu),
                    [yticks[-1]] * len(X_fit_mu),
                    color='gray', alpha=0.35)
    
    ax.set_yticks(yticks)
    ax.set_xticks(X_real)
    ax.set_xticklabels(np.arange(len(X_real)), rotation=45)
    ax.set_xlabel("T")
    if model_name == 'bert': model_name = "bert-base-uncased"
    ax.set_ylabel(r"$\mu_{"+ r"{} / {}".format(dim_i + 1, bert_dim) + r"}$" + " ({})".format(model_name))
    plt.title(fos)
    plt.legend(frameon=False)


#%%
def vector_auto_regression(train_input, train_output, test_output, name):

    def test_model(Y_pred_mu, Y_hat_pred_mu, t):
        eval_train = RMSE_MAE_MAPE(Y_pred_mu[:t, :].T, Y_hat_pred_mu[:t, :].T)
        return eval_train    

    var_mod = dict()
    var_res = dict()
    i = 0
    for fos in tqdm(train_input):
        i += 1
        # 拟合数据
        X_fit_mu  = list()
        X_fit_std = list()
        for t in train_input[fos]:
            X_fit_mu.append(train_input[fos][t][0])
            X_fit_std.append(train_input[fos][t][1])
        for t in train_output[fos]:
            X_fit_mu.append(train_output[fos][t][0])
            X_fit_std.append(train_output[fos][t][1])
        X_fit_mu  = np.array(X_fit_mu)
        X_fit_std = np.array(X_fit_std)
        
        # 预测数据
        Y_pred_mu  = list()
        Y_pred_std = list()
        for t in test_output[fos]:
            Y_pred_mu.append(test_output[fos][t][0])
            Y_pred_std.append(test_output[fos][t][1])
        Y_pred_mu  = np.array(Y_pred_mu)
        Y_pred_std = np.array(Y_pred_std)
        
        # 模型拟合 (VAR)
        model = VAR(endog=np.array(X_fit_mu, dtype=np.float32))
        model_fit = model.fit(trend='n')
        Y_hat_pred_mu = model_fit.forecast(model_fit.endog, steps=len(Y_pred_mu))
        
        # 模型评价 (后续5年/ 7年/ 10年的预测精度)
        eval_test_mu_5  = test_model(Y_pred_mu, Y_hat_pred_mu, 5)
        eval_test_mu_7  = test_model(Y_pred_mu, Y_hat_pred_mu, 7)
        eval_test_mu_10 = test_model(Y_pred_mu, Y_hat_pred_mu, 10)
        
        var_mod[fos] = ""
        var_res[fos] = dict()
        var_res[fos]['TEST_MU'] = np.array([list(eval_test_mu_5), list(eval_test_mu_7), list(eval_test_mu_10)])
        
    return var_res, var_mod


def vector_auto_regression_MP(train_input, train_output, test_output, name, model_name, mp_num=8):
    # 多进程调用vector_auto_regression
    fos_list  = list(train_input.keys())
    total_num = len(fos_list)
    batch_num = math.ceil(total_num/ mp_num)
    start_idx = 0
    end_idx   = 0
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        end_idx = min(start_idx + batch_num, total_num)
        fos_list_i = fos_list[start_idx: end_idx]
        train_input_i  = {fos: train_input[fos]  for fos in fos_list_i} 
        train_output_i = {fos: train_output[fos] for fos in fos_list_i} 
        test_output_i  = {fos: test_output[fos]  for fos in fos_list_i} 
        start_idx = end_idx
        results.append(pool.apply_async(vector_auto_regression, (train_input_i, train_output_i, test_output_i, name,)))
    pool.close()
    pool.join()
    
    var_mod = dict()
    var_res = dict()
    for res in results:
        var_res_i, var_mod_i = res.get()
        for fos in var_res_i: 
            var_mod[fos] = var_mod_i[fos]
            var_res[fos] = var_res_i[fos]
            
    res_save_path = "./Models"
    if not os.path.exists(res_save_path):
        os.mkdir(res_save_path)
    save_file(var_mod, os.path.join(res_save_path, "var_mod({})({}).pkl".format(name, model_name)))
    save_file(var_res, os.path.join(res_save_path, "var_res({})({}).pkl".format(name, model_name)))


def var_plot_case(train_input, train_output, test_output, model_name):
    # 检查vector_auto_regression_MP的预测精度: 预测案例
    
    fos = random.sample(train_input.keys(), 1)[0]
    # 拟合数据
    X_fit_mu  = list()
    X_fit_std = list()
    for t in train_input[fos]:
        X_fit_mu.append(train_input[fos][t][0])
        X_fit_std.append(train_input[fos][t][1])
    for t in train_output[fos]:
        X_fit_mu.append(train_output[fos][t][0])
        X_fit_std.append(train_output[fos][t][1])
    X_fit_mu  = np.array(X_fit_mu)
    X_fit_std = np.array(X_fit_std)
    # 预测数据
    Y_pred_mu  = list()
    Y_pred_std = list()
    for t in test_output[fos]:
        Y_pred_mu.append(test_output[fos][t][0])
        Y_pred_std.append(test_output[fos][t][1])
    Y_pred_mu  = np.array(Y_pred_mu)
    Y_pred_std = np.array(Y_pred_std)
    
    # 模型拟合
    model = VAR(endog=np.array(X_fit_mu, dtype=np.float32))
    model_fit = model.fit(trend='n')
    Y_hat_pred_mu = model_fit.forecast(model_fit.endog, steps=len(Y_pred_mu))
    
    # model = VARMAX(X_fit_mu, order=(1, 1))
    # model_fit = model.fit(disp=False)
    # yhat = model_fit.forecast()

    # 模型评价
    _, bert_dim = X_fit_mu.shape
    r2_mu = list()
    for i in range(bert_dim):
        r2_mu_i = r2_score(Y_pred_mu[:, i], Y_hat_pred_mu[:, i])
        r2_mu.append(max(0, r2_mu_i))
    r2_mu = np.mean(r2_mu)
    dim_i = random.sample(list(np.arange(bert_dim)), 1)[0]
    r2_mu_i = r2_score(Y_pred_mu[:, dim_i], Y_hat_pred_mu[:, dim_i])
    
    # 绘图
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    ax = fig.add_subplot(111)

    Y_real = np.concatenate([X_fit_mu, Y_pred_mu], axis=0)
    Y_real = Y_real[:, dim_i]
    X_real = np.arange(len(Y_real))
    Y_pred = Y_hat_pred_mu[:, dim_i]
    X_pred = np.arange(len(Y_pred)) + len(X_fit_mu)
    
    ax.plot(X_real, Y_real, c='blue', label='Observed', marker='o', markersize=10)
    ax.plot(X_pred, Y_pred,  c='red',  label='Forecast', marker='s', linestyle='--', linewidth=3) # 'Forecast ({:.3f}, {:.3f})'.format(r2_mu_i, r2_mu)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_y   = math.ceil(max(max(Y_real),   max(Y_pred)) * 1000)
    down_y = math.floor( min(min(Y_real), min(Y_pred)) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    
    ax.fill_between(X_real[: len(X_fit_mu)], 
                    [yticks[0]] * len(X_fit_mu),
                    [yticks[-1]] * len(X_fit_mu),
                    color='gray', alpha=0.35)
    
    ax.set_yticks(yticks)
    ax.set_xticks(X_real)
    ax.set_xticklabels(np.arange(len(X_real)), rotation=45)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"${} / {}$".format(dim_i + 1, bert_dim) + " ({})".format(model_name))
    plt.title(fos)
    plt.legend(frameon=False)


def var_predict(train_input, train_output, cut_t, model_name):
    """ 预测向量给解释模型(decoder)解释 """
    pred_vec = dict()
    i = 0
    for fos in tqdm(train_input):
        i += 1
        # 拟合数据
        X_fit_mu  = list()
        X_fit_std = list()
        for t in train_input[fos]:
            X_fit_mu.append(train_input[fos][t][0])
            X_fit_std.append(train_input[fos][t][1])
        for t in train_output[fos]:
            X_fit_mu.append(train_output[fos][t][0])
            X_fit_std.append(train_output[fos][t][1])
        X_fit_mu  = np.array(X_fit_mu)
        X_fit_std = np.array(X_fit_std)
    
        # 模型拟合 (VAR) - forecast in cut_t years
        model_mu       = VAR(endog=np.array(X_fit_mu, dtype=np.float32))
        model_mu_fit   = model_mu.fit(trend='n')
        Y_hat_pred_mu  = model_mu_fit.forecast(model_mu_fit.endog, steps=cut_t)
        
        # 模型拟合 (VAR) - forecast in cut_t years
        model_std      = VAR(endog=np.array(X_fit_std, dtype=np.float32))
        model_std_fit  = model_std.fit(trend='n')
        Y_hat_pred_std = model_std_fit.forecast(model_std_fit.endog, steps=cut_t)
        
        pred_vec[fos]  = Y_hat_pred_mu.T, Y_hat_pred_std.T
    
    save_file(pred_vec, os.path.join(res_explain_path, "pred_vec(var)({})".format(model_name)))
              
              
              
        

#%%
# 神经网络的预测方法
class FeedForward(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.proj1 = nn.Linear(self.input_dim,  self.hidden_dim)
        self.proj2 = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        return self.proj2(torch.relu(self.proj1(x)))


class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size_in, hidden_size_gru, num_embeddings, emb_size, hidden_size_out):
        super(NeuralNetwork, self).__init__()
        
        self.emb_size = emb_size
        self.hidden_size_in  = hidden_size_in
        self.hidden_size_gru = hidden_size_gru
        self.hidden_size_out = hidden_size_out
        
        if self.emb_size > 0:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)    
            self.rnn_layer = nn.LSTM(hidden_size_in + emb_size, hidden_size_gru, batch_first=True)
        else:
            self.rnn_layer = nn.LSTM(hidden_size_in, hidden_size_gru, batch_first=True)
            
        self.in_layer  = FeedForward(input_size,      hidden_size_in,  hidden_size_in)
        self.out_layer = FeedForward(hidden_size_gru, hidden_size_out, hidden_size_out)

    def forward(self, X, X_idx, hidden_h, hidden_c, X_mask=[]):
        
        X_in  = self.in_layer(X)
        
        if self.emb_size > 0:
            X_emb = self.embedding(X_idx)
            batch_size, in_step, _ = X.shape
            X_emb = X_emb.expand(in_step, *X_emb.shape)
            X_emb = X_emb.transpose(0, 1)
            # cat Word Embeddings + Output of FFN
            X_in = torch.cat([X_in, X_emb], axis=-1)
        
        if len(X_mask) != 0:
            # 每个序列长度不一致, 根据X_mask截取获得正确的h_n
            output, (h_n, c_n) = self.rnn_layer(X_in, (hidden_h, hidden_c))
            output_h = list()
            for m, n in enumerate(X_mask):
                output_m_n = output[m: m+1, n-1: n, :]
                output_h.append(output_m_n)
            output_h = torch.cat(output_h, 0)
            h_n = torch.transpose(output_h, 0, 1)
            c_n = h_n
        else:
            output, (h_n, c_n) = self.rnn_layer(X_in, (hidden_h, hidden_c))
            
        Y_out = self.out_layer(output)
        return Y_out, (h_n, c_n)


def train_lstm(vec_dim, train_input, train_output, test_output, fos2idx, name, model_name):
    '''
    Parameters
    ----------
    train_input : TYPE
        DESCRIPTION.
    train_output : TYPE
        DESCRIPTION.
    test_output : TYPE
        DESCRIPTION.
    fos2idx : TYPE
        DESCRIPTION.
    name : string
        mu / std.
    model_name : string
        bert / bert-vae/ doc2vec/ mpnet.
        
    Returns
    -------
    lstm_results : TYPE
        DESCRIPTION.

    '''
    global t0, t1, t2, t3
    global hidden_size_in, hidden_size_gru, emb_size
    
    def sampling_train_batch(train_input, train_output, fos2idx, batch_size, t0, t1, t2):
        """ 随机采用batch_size个主题作为训练数据 """ 
        train_input_len = t1 - t0 + 1
        start_idx = 0
        end_idx   = 0
        fos_list  = list(train_input.keys())
        random.shuffle(fos_list)
        fos_num   = len(fos_list)
        batch_num = math.ceil(fos_num / batch_size)
        
        for i in range(batch_num):
            end_idx   = min(fos_num, start_idx + batch_size)
            batch_fos = fos_list[start_idx: end_idx]
            
            X_mu, X_std, X_p = list(), list(), list()
            Y_mu, Y_std, Y_p = list(), list(), list()
            X_mask = list()
            X_idx  = list()

            for fos in batch_fos:
                X_idx.append(fos2idx[fos])               # 词编号
                # t0 - t1 训练时段
                Ts = train_input[fos].keys()
                min_t, max_t = min(Ts), max(Ts)
                X_mu_fos, X_std_fos, X_p_fos = list(), list(), list()
                for t in range(max(min_t, t0), t1 + 1):  
                    mu_t_i, std_t_i, p_t_i = train_input[fos][t]
                    X_mu_fos.append(mu_t_i)
                    X_std_fos.append(std_t_i)
                    X_p_fos.append(p_t_i)
                # 补齐输入 (padding)
                X_mask.append(len(X_mu_fos))             # 非补齐的输入长度
                while len(X_mu_fos) < train_input_len:
                    X_mu_fos.insert(-1,  X_mu_fos[-1])   # 末尾补数
                    X_std_fos.insert(-1, X_std_fos[-1])  # 末尾补数
                    X_p_fos.insert(-1, 0)                # 末尾补数
                X_mu.append(X_mu_fos)
                X_std.append(X_std_fos)
                X_p.append(X_p_fos)
                
                # t1 - t2 预测时段
                Y_mu_fos, Y_std_fos, Y_p_fos = list(), list(), list()
                for t in range(t1 + 1, t2 + 1):
                    mu_t_i, std_t_i, p_t_i = train_output[fos][t]
                    Y_mu_fos.append(mu_t_i)
                    Y_std_fos.append(std_t_i)
                    Y_p_fos.append(p_t_i)
                Y_mu.append(Y_mu_fos)
                Y_std.append(Y_std_fos)
                Y_p.append(Y_p_fos) 
                
            X_mu, X_std, X_p = np.array(X_mu), np.array(X_std), np.array(X_p)[:, :, np.newaxis]
            Y_mu, Y_std, Y_p = np.array(Y_mu), np.array(Y_std), np.array(Y_p)[:, :, np.newaxis]
            X_mask = np.array(X_mask)
            X_idx  = np.array(X_idx)
            yield (X_mu, Y_mu), (X_std, Y_std), (X_p, Y_p), X_mask, X_idx, batch_fos
            start_idx = end_idx
    
    
    # 创建模型
    enc_mu  = NeuralNetwork(vec_dim, hidden_size_in, hidden_size_gru, len(fos2idx), emb_size, vec_dim)
    dec_mu  = NeuralNetwork(vec_dim, hidden_size_in, hidden_size_gru, len(fos2idx), -1, vec_dim)
    enc_mu  = enc_mu.to(device)
    dec_mu  = dec_mu.to(device)
    # loss
    loss_mse = nn.HuberLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(list(enc_mu.parameters()) + list(dec_mu.parameters()), lr=learning_rate)
                                 
    pirnt_loss = list()
    epoch = 250
    batch_size = 64
    for i in range(epoch):
        pirnt_loss_i = list()
        train_data_iter = sampling_train_batch(train_input, train_output, fos2idx, batch_size, t0, t1, t2)
        for j, batch_data in enumerate(train_data_iter):

            enc_mu.train()
            dec_mu.train()
            optimizer.zero_grad()
            
            (X_mu, Y_mu), (X_std, Y_std), (_, _), X_mask, X_idx, batch_fos = batch_data
            # X_mu = np.log(np.maximum(X_mu, 1e-5))
            # Y_mu = np.log(np.maximum(Y_mu, 1e-5))

            if name == "mu":
                X = X_mu
                Y = Y_mu
            if name == "std":
                X = X_std
                Y = Y_std
            X = torch.tensor(X, dtype=torch.float32).to(device)
            Y = torch.tensor(Y, dtype=torch.float32).to(device)
            X_idx = torch.tensor(X_idx).to(device)
            enc_hid = torch.tensor(np.zeros((1, len(X), hidden_size_gru)), dtype=torch.float32).to(device)
   
            # ENCODER
            enc_mu_out, (enc_mu_h, enc_mu_c) = enc_mu(X, X_idx, enc_hid, enc_hid, X_mask) # X_mask
            # DECODER
            max_length = Y.shape[1]
            teacher_forcing_ratio = 0.5
            use_teacher_forcing   = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                dec_mu_out = list()
                dec_mu_h = enc_mu_h
                dec_mu_c = enc_mu_c
                for l in range(max_length):
                    if l == 0:
                        dec_mu_in = X[:, -1:, :]
                    else:
                        dec_mu_in = Y[:, l-1: l, :]
                    dec_mu_out_l, (dec_mu_h, dec_mu_c) = dec_mu(dec_mu_in, X_idx, dec_mu_h, dec_mu_c)
                    dec_mu_out.append(dec_mu_out_l)
                dec_mu_out = torch.cat(dec_mu_out, axis=1)       
            else:
                # Without teacher forcing: use its own predictions as the next input
                dec_mu_out = list()
                dec_mu_h = enc_mu_h
                dec_mu_c = enc_mu_c    
                for l in range(max_length):
                    if l == 0:
                        dec_mu_in = X[:, -1:, :]
                    else:
                        dec_mu_in = dec_mu_out_l.detach()  # detach from history as input
                    dec_mu_out_l, (dec_mu_h, dec_mu_c) = dec_mu(dec_mu_in, X_idx, dec_mu_h, dec_mu_c)
                    dec_mu_out.append(dec_mu_out_l)
                dec_mu_out = torch.cat(dec_mu_out, axis=1)
            
            loss_mse_mu = loss_mse(dec_mu_out, Y)
            loss_mse_mu.backward()
            torch.nn.utils.clip_grad_norm_(enc_mu.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(dec_mu.parameters(), 5)
            optimizer.step()
            
            torch.cuda.empty_cache()
            print("Epoch {} - Batch {} : {:.8f}".format(i + 1, j + 1, loss_mse_mu))
            pirnt_loss_i.append(loss_mse_mu.detach().cpu().numpy()) 
        pirnt_loss.append(np.mean(pirnt_loss_i))
        
    # 绘制 LOSS 曲线name
    plt.plot(pirnt_loss)
    save_file(enc_mu, "./Models/enc({})({}).pkl".format(name, model_name))
    save_file(dec_mu, "./Models/dec({})({}).pkl".format(name, model_name))


def test_lstm(vec_dim, train_input, train_output, test_output, fos2idx, name, model_name):
    
    global t0, t1, t2, t3
    global hidden_size_in, hidden_size_gru
    
    def plot_case(X, X_mask, Y, dec_mu_out, batch_fos, i=0):
        # case study
        fos = batch_fos[i]
        dim_idx = random.sample(list(np.arange(0, dec_mu_out.shape[-1])), 1)[0]
        y_know  = X[i, :, dim_idx].detach().cpu().numpy()       # input
        # 剔除mask
        if len(y_know) == t1-t0+1:
            y1 = y_know[: (t1-t0+1)]
            y1 = y1[: X_mask[i]]
            y_know = y1
        else:
            y1 = y_know[: (t1-t0+1)]
            y1 = y1[: X_mask[i]]
            y2 = y_know[(t1-t0+1):]
            y_know = np.concatenate([y1, y2])
        # 
        y_pred  = dec_mu_out[i, :, dim_idx].detach().cpu().numpy()
        y_real  = Y[i, :, dim_idx].detach().cpu().numpy()
        y1 = np.concatenate([y_know, y_pred])
        y2 = np.concatenate([y_know, y_real])
        plt.plot(np.arange(len(y1)), y1, label='Pred', linestyle='--')
        plt.plot(np.arange(len(y2)), y2, label='Real', c='gray')
        plt.ylabel("{} / {}".format(dim_idx + 1, dec_mu_out.shape[-1]))
        plt.title(fos)
        plt.legend()
    
    def sampling_test_batch(train_input, train_output, test_output, fos2idx, batch_size, t0, t1, t2, t3):
        """ 随机采用batch_size个主题作为预测数据 """ 
        train_input_len = t2 - t0 + 1
        start_idx = 0
        end_idx   = 0
        all_fos   = list(train_input.keys())
        batch_num = math.ceil(len(all_fos) / batch_size)
        for i in range(batch_num):
            end_idx = min(len(all_fos), start_idx + batch_size)
            batch_fos = all_fos[start_idx: end_idx]
            
            X_mu, X_std, X_p = list(), list(), list()
            Y_mu, Y_std, Y_p = list(), list(), list()
            X_mask = list()
            X_idx  = list()
            for fos in batch_fos:
                X_idx.append(fos2idx[fos])
                # t0 - t1 训练时段
                Ts = train_input[fos].keys()
                min_t, max_t = min(Ts), max(Ts)
                X_mu_fos, X_std_fos, X_p_fos = list(), list(), list()
                for t in range(max(min_t, t0), t1 + 1):
                    mu_t_i, std_t_i, p_t_i = train_input[fos][t]
                    X_mu_fos.append(mu_t_i)
                    X_std_fos.append(std_t_i)
                    X_p_fos.append(p_t_i)
                # t1 - t2 训练时段
                for t in range(t1 + 1, t2 + 1):
                    mu_t_i, std_t_i, p_t_i = train_output[fos][t]
                    X_mu_fos.append(mu_t_i)
                    X_std_fos.append(std_t_i)
                    X_p_fos.append(p_t_i)
                # 补齐输入 (padding)
                X_mask.append(len(X_mu_fos))
                while len(X_mu_fos) < train_input_len:
                    X_mu_fos.insert(-1,  X_mu_fos[-1])
                    X_std_fos.insert(-1, X_std_fos[-1])
                    X_p_fos.insert(-1, 0)
                X_mu.append(X_mu_fos)
                X_std.append(X_std_fos)
                X_p.append(X_p_fos) 
                
                # t2 - t3 预测时段
                Y_mu_fos, Y_std_fos, Y_p_fos = list(), list(), list()
                for t in range(t2 + 1, t3 + 1):
                    mu_t_i, std_t_i, p_t_i = test_output[fos][t]
                    Y_mu_fos.append(mu_t_i)
                    Y_std_fos.append(std_t_i)
                    Y_p_fos.append(p_t_i)
                Y_mu.append(Y_mu_fos)
                Y_std.append(Y_std_fos)
                Y_p.append(Y_p_fos) 
                
            X_mu, X_std, X_p = np.array(X_mu), np.array(X_std), np.array(X_p)[:, :, np.newaxis]
            Y_mu, Y_std, Y_p = np.array(Y_mu), np.array(Y_std), np.array(Y_p)[:, :, np.newaxis]
            X_mask = np.array(X_mask)
            X_idx  = np.array(X_idx)
            yield (X_mu, Y_mu), (X_std, Y_std), (X_p, Y_p), X_mask, X_idx, batch_fos
            print("{} / {}".format(i, batch_num))
            start_idx = end_idx
    
    def test_model(Y_pred_mu, Y_hat_pred_mu, t):
        eval_train = RMSE_MAE_MAPE(Y_pred_mu[:t, :].T, Y_hat_pred_mu[:t, :].T)
        return eval_train    

    # 读取模型
    # name = 'mu'
    enc_mu = read_file("./Models/enc({})({}).pkl".format(name, model_name))
    dec_mu = read_file("./Models/dec({})({}).pkl".format(name, model_name))
    enc_mu.eval()
    dec_mu.eval()

    # 读取数据
    lstm_res = dict()
    batch_size = 128
    test_data_iter = sampling_test_batch(train_input, train_output, test_output, fos2idx, batch_size, t0, t1, t2, t3)
    for batch_data in test_data_iter:
        (X_mu, Y_mu), (X_std, Y_std), (X_p, Y_p), X_mask, X_idx, batch_fos = batch_data
        if name == "mu":
            X = X_mu
            Y = Y_mu
        if name == "std":
            X = X_std
            Y = Y_std
        X = torch.tensor(X, dtype=torch.float32).to(device)
        Y = torch.tensor(Y, dtype=torch.float32).to(device)
        X_idx = torch.tensor(X_idx).to(device)
        enc_hidden = torch.tensor(np.zeros((1, len(X), hidden_size_gru)), dtype=torch.float32).to(device)

        # ENCODER
        enc_mu_out, (enc_mu_h, enc_mu_c) = enc_mu(X, X_idx, enc_hidden, enc_hidden, X_mask)
        # DECODER
        max_length = Y.shape[1]
        dec_mu_out = list()
        dec_mu_h = enc_mu_h
        dec_mu_c = enc_mu_c
        if False:
            # Teacher forcing: Feed the target as the next input
            dec_mu_out = list()
            dec_mu_h = enc_mu_h
            dec_mu_c = enc_mu_c
            for l in range(max_length):
                if l == 0:
                    dec_mu_in = X[:, -1:, :]
                else:
                    dec_mu_in = Y[:, l-1: l, :]
                dec_mu_out_l, (dec_mu_h, dec_mu_c) = dec_mu(dec_mu_in, X_idx, dec_mu_h, dec_mu_c)
                dec_mu_out.append(dec_mu_out_l)
            dec_mu_out = torch.cat(dec_mu_out, axis=1)       
        else:
            # Without teacher forcing: use its own predic tions as the next input
            dec_mu_out = list()
            dec_mu_h = enc_mu_h
            dec_mu_c = enc_mu_c    
            for l in range(max_length):
                if l == 0:
                    dec_mu_in = X[:, -1:, :]
                else:
                    dec_mu_in = dec_mu_out_l.detach()  # detach from history as input
                dec_mu_out_l, (dec_mu_h, dec_mu_c) = dec_mu(dec_mu_in, X_idx, dec_mu_h, dec_mu_c)
                dec_mu_out.append(dec_mu_out_l)
            dec_mu_out = torch.cat(dec_mu_out, axis=1)
        
        # 逐个fos计算MSE
        for i, fos in enumerate(batch_fos):
            Y_pred_mu       = Y.detach().cpu().numpy()[i, :, :]
            Y_hat_pred_mu   = dec_mu_out.detach().cpu().numpy()[i, :, :]
            
            eval_test_mu_5  = test_model(Y_pred_mu, Y_hat_pred_mu, 5)
            eval_test_mu_7  = test_model(Y_pred_mu, Y_hat_pred_mu, 7)
            eval_test_mu_10 = test_model(Y_pred_mu, Y_hat_pred_mu, 10)
            lstm_res[fos]   = np.array([list(eval_test_mu_5), list(eval_test_mu_7), list(eval_test_mu_10)])
        
            # plot_case(X, X_mask, Y, dec_mu_out, batch_fos, i)
    save_file(lstm_res, "./Models/lstm_res({})({}).pkl".format(name, model_name))


#%%
def delete_outlier(lr_mapes):
    lr_mapes = np.array(lr_mapes)
    # IQR
    Q1 = np.percentile(lr_mapes, 25, interpolation = 'midpoint')
    Q3 = np.percentile(lr_mapes, 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    # Above Upper bound
    upper = lr_mapes >= (Q3+1.5*IQR)
    # Below Lower bound
    lower = lr_mapes <= (Q1-1.5*IQR)
    lr_mapes[upper] = np.nan
    lr_mapes[lower] = np.nan
    return lr_mapes


def generate_prediction_data(data_file_path, model_name, filtered_nop):
    """ 生成预测模型所需要的数据 """
    # 主题逐年的向量均值和标准差数据
    save_file_path       = os.path.join(data_file_path, model_name)
    # fos_mu_std_file_path = os.path.join(save_file_path, "fos_mu_std_nop({}).pkl".format(False))
    fos_mu_std_file_path = os.path.join(save_file_path, "fos_mu_std_nop.pkl")
    if os.path.exists(fos_mu_std_file_path):
        fos_mu_std_nop = read_file(fos_mu_std_file_path)
    else:
        print("需要由{}生成文件".format("Semantic_Movement_Verify.py"))        
    
    # 挑选累计发文量超过或等于100的主题词
    fos_mu_std_nop2 = dict()
    for fos in fos_mu_std_nop:
        if len(fos_mu_std_nop[fos].keys()) == 0:
            continue
        
        end_t = max(fos_mu_std_nop[fos].keys())
        acc_nop_t = fos_mu_std_nop[fos][end_t][-1]
        if acc_nop_t >= filtered_nop:
            fos_mu_std_nop2[fos] = fos_mu_std_nop[fos]
    print("过滤频率低于{}得到的主题数目{} / {}".format(filtered_nop, len(fos_mu_std_nop2), len(fos_mu_std_nop)))
    
    def create_train_data(fos_mu_std_nop):
        # 统计词频
        yearlynop = dict()
        for fos in fos_mu_std_nop:
            for t in fos_mu_std_nop[fos]:
                mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t]
                if t not in yearlynop:
                    yearlynop[t] = nop_t_i_yearly
                else:
                    yearlynop[t] += nop_t_i_yearly
                   
        # 预测时间段划分, 训练集 验证集 测试集划分
        train_input  = dict()
        train_output = dict()
        # test_input = train_input + train_ouput
        test_output  = dict()  
        fos2idx      = dict()
        for idx, fos in enumerate(fos_mu_std_nop):
            fos2idx[fos] = idx
            Ts = fos_mu_std_nop[fos].keys()
            min_t, max_t = min(Ts), max(Ts)
            if min_t <= t1:
                # before t1 来构建 reference vocabulary
                train_input[fos]  = dict()
                train_output[fos] = dict()
                test_output[fos]  = dict()
                # [t0, t1]
                for t in range(max(min_t, t0), t1 + 1):
                    # t年无数据, 则考虑t-1年
                    if t not in fos_mu_std_nop[fos]:
                        t_ = t - 1
                        while True:
                            if t_ in fos_mu_std_nop[fos]:
                                mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t_]
                                break
                            else:
                                t_ = t_ - 1
                    else:
                        mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t]
                    train_input[fos][t] = (np.array(mu_t_i,  np.float16), 
                                           np.array(std_t_i, np.float16), 
                                           nop_t_i_yearly / yearlynop[t])
                # [t1+1, t2]
                for t in range(t1 + 1, t2 + 1):          
                    if t not in fos_mu_std_nop[fos]:
                        t_ = t - 1
                        while True:
                            if t_ in fos_mu_std_nop[fos]:
                                mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t_]
                                break
                            else:
                                t_ = t_ - 1
                    else:
                        mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t]
                    train_output[fos][t] = (np.array(mu_t_i,  np.float16), 
                                            np.array(std_t_i, np.float16), 
                                            nop_t_i_yearly / yearlynop[t])
                # [t2+1, t3]
                for t in range(t2 + 1, t3 + 1):          
                    if t not in fos_mu_std_nop[fos]:
                        t_ = t - 1
                        while True:
                            if t_ in fos_mu_std_nop[fos]:
                                mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t_]
                                break
                            else:
                                t_ = t_ - 1
                    else:
                        mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t]
                    test_output[fos][t] = (np.array(mu_t_i,  np.float16), 
                                           np.array(std_t_i, np.float16), 
                                           nop_t_i_yearly / yearlynop[t])
            else:
                continue
        return train_input, train_output, test_output, fos2idx
        
    train_input, train_output, test_output, fos2idx = create_train_data(fos_mu_std_nop2)
    print("参与训练集的主题数目: {} / {}".format(len(train_input), len(fos_mu_std_nop2))) 
    return train_input, train_output, test_output, fos2idx, fos_mu_std_nop, fos_mu_std_nop2


def show_data_case(train_input, train_output, test_output, fos=''):
    # 展示轨道, 简单观测是否可预测
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    if len(fos) == 0:
        fos = random.sample(train_input.keys(), 1)[0]

    for t in train_input[fos]:
        mu_t_i = train_input[fos][t][0]
        break
        
    dim_idx = random.sample(list(np.arange(0, mu_t_i.shape[0])), 1)[0]
    obs_idx = 0  # 0: mu, 1: std, 2: p_t_i
    X1  = list(train_input[fos].keys())
    Y1  = [train_input[fos][x][obs_idx][dim_idx]  for x in X1]
    X2  = list(train_output[fos].keys())
    Y2  = [train_output[fos][x][obs_idx][dim_idx] for x in X2]
    X3  = list(test_output[fos].keys())
    Y3  = [test_output[fos][x][obs_idx][dim_idx]  for x in X3]
    plt.plot(X1 + X2 + X3, Y1 + Y2 + Y3, c='gray')
    plt.plot(X1, Y1, c='blue',  linestyle='--', linewidth=3)
    plt.plot(X2, Y2, c='green', linestyle='--', linewidth=3)
    plt.plot(X3, Y3, c='red',   linestyle='--', linewidth=3)
    plt.xticks(X1 + X2 + X3, rotation=45, fontsize=15)
    plt.xlabel("Time")
    plt.ylabel("{} / {}".format(dim_idx + 1, mu_t_i.shape[0]))
    plt.title(fos)


def evaluate_accuracy(lr_res, var_res, lstm_res):
    # 预测精度作图评价
    def calmean(seq):
        return "{:.4f}".format(np.nanmedian(seq))
    
    tb = pt.PrettyTable()
    tb.field_names = ["Metrics", "Time", "LR", "VAE", "LSTM"]
   
    data = dict()
    rmses, maes, mapes = list(), list(), list()
    models = list()
    Ts     = list()
    var_res_id = {5: 0, 7: 1, 10: 2}
    for t in [5, 7, 10]:
        lr_rmses,   lr_maes,   lr_mapes   = list(), list(), list()
        vae_rmses,  vae_maes,  vae_mapes  = list(), list(), list()
        lstm_rmses, lstm_maes, lstm_mapes = list(), list(), list()
        for fos in var_res:
            # LR
            rmse, mae, mape = lr_res[fos]['TEST_MU'][var_res_id[t]]
            lr_rmses.append(rmse)
            lr_maes.append(mae)
            lr_mapes.append(mape)
            # VAR
            rmse, mae, mape = var_res[fos]['TEST_MU'][var_res_id[t]]
            vae_rmses.append(rmse)
            vae_maes.append(mae)
            vae_mapes.append(mape)
            # LSTM
            rmse, mae, mape = lstm_res[fos][var_res_id[t]]
            lstm_rmses.append(rmse)
            lstm_maes.append(mae)
            lstm_mapes.append(mape)
            
        lr_rmses   = delete_outlier(lr_rmses)
        vae_rmses  = delete_outlier(vae_rmses)
        lstm_rmses = delete_outlier(lstm_rmses)
        
        lr_maes    = delete_outlier(lr_maes)
        vae_maes   = delete_outlier(vae_maes)
        lstm_maes  = delete_outlier(lstm_maes)
        
        lr_mapes   = delete_outlier(lr_mapes)
        vae_mapes  = delete_outlier(vae_mapes)
        lstm_mapes = delete_outlier(lstm_mapes)
    
        tb.add_row(["RMSE", t, calmean(lr_rmses), calmean(vae_rmses), calmean(lstm_rmses)])
        tb.add_row(["MAE",  t, calmean(lr_maes),  calmean(vae_maes),  calmean(lstm_maes)])
        tb.add_row(["MAPE", t, calmean(lr_mapes), calmean(vae_mapes), calmean(lstm_mapes)])
        tb.add_row(["-", "-", "-", "-", "-"])
    
        rmses_t = np.concatenate([lr_rmses, vae_rmses, lstm_rmses])
        mase_t  = np.concatenate([lr_maes,  vae_maes,  lstm_maes])
        mape_t  = np.concatenate([lr_mapes, vae_mapes, lstm_mapes]) 
        model_t = len(lr_mapes) * ['LR'] + len(vae_mapes) * ['VAR'] + len(lstm_mapes) * ['LSTM']
        ts      = len(lr_mapes) * [t] + len(vae_mapes) * [t] + len(lstm_mapes) * [t]
        
        rmses.append(rmses_t)
        maes.append(mase_t)
        mapes.append(mape_t)
        models += model_t
        Ts += ts
    print(tb)
    
    data['RMSE']  = np.concatenate(rmses)
    data['MAE']   = np.concatenate(maes)
    data['MAPE']  = np.concatenate(mapes)
    data['Model'] = models
    data['t']     = Ts
    pdata = pd.DataFrame(data)
    
    return pdata


#%%
if __name__ == "__main__":
    #  Section 3.3 Predicting the motion of topic emebeddings
    t0 = 1980
    t1 = 1985 # 训练时输入 - 确定 reference vocabulary
    t2 = 1990 # 训练时预测 - 预测时输入
    t3 = 2000 # 预测时输出
    
    func_type = 'y=ax+b'
    name = 'mu'
    
    # LSTM 参数配置
    input_size_dict = {
        "bert_vae": 256, 
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "bert": 768
    }
    emb_size = 128
    hidden_size_in  = 512
    hidden_size_gru = 512    
    
    def main():
        # all-MiniLM-L6-v2 # bert # bert_vae # doc2vec # "all-mpnet-base-v2"
        model_name_list = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "bert"]
        # model_name = "all-mpnet-base-v2"
        for model_name in model_name_list:
            # 准备预测数据
            filtered_nop2 = 500
            train_input, train_output, test_output, fos2idx, fos_mu_std_nop, fos_mu_std_nop2 = generate_prediction_data(data_file_path, model_name, filtered_nop2)
            show_data_case(train_input, train_output, test_output)
            
            # linear model
            train_lr_MP(train_input, train_output, test_output, model_name, func_type)
            test_lr_MP(train_input, train_output, test_output, model_name, func_type)
            # vector autoregression     
            vector_auto_regression_MP(train_input, train_output, test_output, name, model_name)
            # LSTM
            vec_dim = input_size_dict[model_name] # embedding的维度
            train_lstm(vec_dim, train_input, train_output, test_output, fos2idx, name, model_name)
            test_lstm(vec_dim, train_input,  train_output, test_output, fos2idx, name, model_name)
    
        """ 绘制预测结果的RMSE, MAE, MAPE """
        lr_res   = read_file("./Models/lr_res({})({}).pkl".format(model_name, func_type))
        var_res  = read_file("./Models/var_res({})({}).pkl".format(name, model_name))
        lstm_res = read_file("./Models/lstm_res({})({}).pkl".format(name, model_name))
        pdata = evaluate_accuracy(lr_res, var_res, lstm_res)    
        
        fig = plt.figure(figsize=(8, 6))
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        config = {
                  "font.family" : "Times New Roman",
                  "font.size" : 20
                  }
        rcParams.update(config)
        
        sns.violinplot(x="Model", y="RMSE", hue='t', data=pdata, showextrema=False)
        plt.title(model_name)
        plt.legend(frameon=False, loc='upper left')
        # yticks = np.arange(0, 400, 50)
        # ylabels = ["{}%".format(y) for y in yticks] 
        # plt.yticks(yticks, labels=ylabels)
        # plt.yticks(yticks)
        
        """ 案例分析 """
        lr_plot_case(train_input, train_output, test_output, model_name)
        var_plot_case(train_input, train_output, test_output, model_name)
