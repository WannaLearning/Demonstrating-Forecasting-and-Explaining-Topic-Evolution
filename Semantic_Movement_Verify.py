#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:23:33 2023

@author: aixuexi
"""
import os 
import re
import json
import math
import time
import random
import pickle
import string
import sklearn
import multiprocessing
import pandas as pd
import seaborn as sns 
import autograd.numpy as np
import prettytable as pt
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from tqdm import tqdm
from sklearn import decomposition
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

from Extract_cs_data import *


#%%
def calculate_accumulative_mu_std(filtered_fos, vec_file_path, bAccumulative):
    # 计算每个主题的逐年累向量的均值和标准差
    fos_mu_std_nop = dict()
    for fos in tqdm(filtered_fos):
        dic = read_file(os.path.join(vec_file_path, fos + ".pkl"))
        fos_mu_std_nop[fos] = dict()
        Ts = list(sorted(dic.keys()))
        min_t = min(Ts)
        nop_t_i = 0
        for t_i in Ts:
            vec_arr = list()
            if bAccumulative:
                # 累计 [min_t, t_i]
                for t_j in range(min_t, t_i + 1):          
                    if t_j in dic:
                        vec_arr_j = dic[t_j]
                        vec_arr  += vec_arr_j
                    if t_j == t_i:
                        # 第ti年的频率
                        if t_i in dic:
                            nop_t_i_yearly = len(dic[t_i])
                        else:
                            nop_t_i_yearly = 0     
                vec_arr = np.array(vec_arr, dtype=np.float64)
                
                if len(vec_arr) > 10: 
                    mu_t_i  = np.mean(vec_arr, axis=0)     # 累计向量的均值
                    std_t_i = np.std(vec_arr,  axis=0)     # 累计向量的标准差
                    mu_t_i  = np.array(mu_t_i,  dtype=np.float32)
                    std_t_i = np.array(std_t_i, dtype=np.float32)
                    nop_t_i += nop_t_i_yearly              # 累计发文量
                    fos_mu_std_nop[fos][t_i] = (mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i)  
            else:
                # 非累计, t_i年份
                if t_i in dic:
                   vec_arr = dic[t_i]
                nop_t_i_yearly = len(vec_arr)
                
                if len(vec_arr) > 10: 
                   mu_t_i  = np.mean(vec_arr, axis=0)      # 累计向量的均值
                   std_t_i = np.std(vec_arr,  axis=0)      # 累计向量的标准差
                   mu_t_i  = np.array(mu_t_i,  dtype=np.float32)
                   std_t_i = np.array(std_t_i, dtype=np.float32)
                   nop_t_i += nop_t_i_yearly               # 累计发文量
                   fos_mu_std_nop[fos][t_i] = (mu_t_i, std_t_i, nop_t_i_yearly, nop_t_i)           
    return fos_mu_std_nop


def calculate_accumulative_mu_std_MP(filtered_fos, vec_file_path, bAccumulative, mp_num=8):
    # 多进程处理 - calculate_accumulative_mu_std FUNCTION
    filtered_fos_list = list(filtered_fos.keys())
    total_num = len(filtered_fos_list)
    batch_num = math.ceil(total_num/ mp_num)
    start_idx = 0
    end_idx   = 0
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        end_idx = min(start_idx + batch_num, total_num)
        filtered_fos_i = filtered_fos_list[start_idx: end_idx]
        start_idx = end_idx
        results.append(pool.apply_async(calculate_accumulative_mu_std, (filtered_fos_i, vec_file_path, bAccumulative)))
    pool.close()
    pool.join()
    # 合并结果 
    fos_mu_std_nop = dict()
    for res in results:
        fos_mu_std_nop_i = res.get()
        for fos in fos_mu_std_nop_i:
            fos_mu_std_nop[fos] = fos_mu_std_nop_i[fos]
    return fos_mu_std_nop
    

#%%
def plot_verify_1d_movement_func(fos, fos_mu_std_nop, model_name, dim=-1):
    """
        观察单个主题在向量空间中移动 
        观察一个向量维度的均值变化
    """
    def lienar_func(x, a, b, c, d):
         y = np.multiply(x ** 3, a) + np.multiply(x ** 2, b) + np.multiply(x, c) + d
         return y
     
    global T0_, T2_
     
    if dim < 0:
        # 确定embedding_mu的维度, 然后随机挑选一维可视化
        for fos_ in fos_mu_std_nop:
            for t in fos_mu_std_nop[fos_]:
                vec_dim = fos_mu_std_nop[fos_][t][0].shape[0]
                break
            break
        D_samples = np.random.multinomial(n=1, pvals=[1/vec_dim] * vec_dim)
        dim = np.arange(vec_dim)[D_samples > 0][0]
        
    years = list(fos_mu_std_nop[fos].keys())
    years = sorted(years)
    years = np.array(years)
    
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 25
              }
    rcParams.update(config)
    
    label_fs = 25
    lw = 3
    ax1 = fig.add_subplot(111)
    X = list()
    Y = list()
    for t in years:
        if t > T2_ or t < T0_:
            continue
        mu_t, std_t, nop_t_i_yearly, nop_t_i = fos_mu_std_nop[fos][t]
        Y.append(mu_t[dim])
        X.append(t)
    ax1.plot(X, Y, c='red', marker='*', markersize=15, linewidth=lw)
    
    X = np.array(X)
    X_normal = (X - min(X)) / (max(X) - min(X))
    popt_Xd, pcov_Xd = curve_fit(lienar_func, X_normal, Y)
    Y_pred = lienar_func(X_normal, *popt_Xd)
    
    label = ''
    for i, coef in enumerate(popt_Xd):
        if i == 0:
            label += "{:.2f}".format(coef)
            label += r'$t^3$'
        elif i == 1:
            if coef > 0:
                label += "+{:.2f}".format(coef)
            else:
                label += "-{:.2f}".format(abs(coef))
            label += r'$t^2$'
        elif i == 2:
            if coef > 0:
                label += "+{:.2f}".format(coef)
            else:
                label += "-{:.2f}".format(abs(coef))
            label += r'$t$'
        else:
            if coef > 0:
                label += "+{:.2f}".format(coef)
            else:
                label += "-{:.2f}".format(abs(coef))
                
    ax1.plot(X, Y_pred, c='black', linestyle='--', linewidth=lw, label=label)
    ax1.legend(fontsize=label_fs)
    ax1.set_xlabel("Time ({})".format(model_name))
    ax1.set_ylabel(r"$\mu$ ({}th in {})".format(dim, vec_dim))
    up_x   = max(Y)
    up_x   = math.ceil(up_x   * 10000)
    down_x = min(Y)
    down_x = math.ceil(down_x * 10000)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    yticks = np.linspace(down_x, up_x, 5) / 10000
    ax1.set_yticks(yticks)
    plt.title(fos, fontsize=label_fs, fontweight='bold')
    plt.tight_layout()


def plot_verify_2d_movement_func(fos, vec_file_path, samples_num=100):
    """
        观察单个主题在向量空间中移动 
        逐年散点2D图
    """
    global T0_, T2_
    model_name = vec_file_path.split('/')[-2]
    dic  = read_file(os.path.join(vec_file_path, fos)) # fos的向量字典
    beg_year = max(min(dic.keys()), T0_)               # fos的启始年
    end_year = min(max(dic.keys()), T2_)               # fos的末年
    vecs = get_vec_func(dic, beg_year, end_year)       # beg_year - end_year 所有向量
    
    # pca reduce dim
    pca = decomposition.PCA(n_components=2)            # pca 
    pca.fit(vecs)
    
    acc_nop = 0
    vecs_2d_ts = dict()
    for t in range(beg_year, end_year + 1):
        if t > T2_ or t < T0_:
            continue
        vecs_t = get_vec_func(dic, t, t)
        acc_nop += len(vecs_t)
        if acc_nop >= 10:
            if len(vecs_t) != 0:    
                vecs_2d_t = pca.transform(vecs_t)
                vecs_2d_ts[t] = vecs_2d_t
            else:
                vecs_2d_ts[t] = []         
    ts     = vecs_2d_ts.keys()
    beg_t  = min(ts)                               # fos的启始年
    end_t  = max(ts)                               # fos的最后一年 2018         
    dates  = np.arange(beg_t, end_t + 1)           # Time
    colors = np.linspace(-1, 1, len(dates))        # color bar 
    middle_t = int((beg_t + end_t) / 2)
    
    # 绘图
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    ax = fig.add_subplot(111)
    
    xyzcs = list()  # (x, y, z, color)
    cents = list()
    svc_y = list()
    for i, t in enumerate(range(beg_t, end_t + 1)):
        vecs_2d_t = vecs_2d_ts[t]
        if len(vecs_2d_t) == 0:
            continue
        
        samples_size = min(len(vecs_2d_t), samples_num)
        print("{} : {}/{}".format(t, samples_size, len(vecs_2d_t)))
        vecs_2d_t = random.sample(list(vecs_2d_t), samples_size)
        vecs_2d_t = np.array(vecs_2d_t)
        xs = vecs_2d_t[:, 0]
        ys = vecs_2d_t[:, 1]
        cs = colors[i] * np.ones(len(xs))
        xyzcs.append(np.array([xs, ys, cs]))
        # cents.append(np.array([np.mean(xs), np.mean(ys), colors[i]]))
        
        # (before middle_t 的是0类, after middle_t 是1类)
        if t > middle_t:
            svc_y += [1] * len(xs)
        else:
            svc_y += [0] * len(xs)

    cmap  = 'seismic'
    xyzcs = np.concatenate(xyzcs, axis=-1)  
    ax.scatter(xyzcs[0, :], xyzcs[1, :], c=xyzcs[2, :], cmap=cmap, s=4, alpha=1)
    
    # 轴刻度
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_x   = math.ceil(max(xyzcs[0, :]) * 1000)
    down_x = math.floor(min(xyzcs[0, :]) * 1000)
    xticks = np.linspace(down_x, up_x, 5) / 1000
    ax.set_xticks(xticks)
    
    up_y   = math.ceil(max(xyzcs[1, :]) * 1000)
    down_y = math.floor(min(xyzcs[1, :]) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    ax.set_yticks(yticks)
    plt.title(fos[:-4], fontsize=20, fontweight='bold')
    
    # plot hypeplane (before middle year 的是0类, after middle year 是1类)
    svc = svm.SVC(kernel='linear', C=1, class_weight='balanced')
    svc.fit(xyzcs[:2, :].T, svc_y)
    svc_y_pred = svc.predict(xyzcs[:2, :].T)
    # 准确率, 召回率, 宏f1
    svc_res = sklearn.metrics.classification_report(svc_y, svc_y_pred, output_dict=True, zero_division=0)
    pre = svc_res['macro avg']['precision']
    rec = svc_res['macro avg']['recall']
    f1  = svc_res['macro avg']['f1-score']
    # 超平面
    w = svc.coef_[0]
    k = - w[0] / w[1]
    b = - svc.intercept_[0] / w[1]
    plane_x_ = np.linspace(min(xticks), max(xticks))
    plane_x  = list()
    plane_y  = list()
    for x in plane_x_:
        y = k * x + b
        if y >= yticks[0] and y <= yticks[-1]:
            plane_x.append(x)
            plane_y.append(y)
    ax.plot(plane_x, plane_y, c='black', linewidth=2, linestyle='--', label=r"Hyperplane: 0 ($\leq${}), 1 ($>${})".format(middle_t, middle_t))
    # 法向量
    Ax = np.median(plane_x)
    Ay = k * Ax + b
    Bx = np.percentile(plane_x, 75)
    By = (-1 / k) * (Bx - Ax) + Ay
    if svc.predict(np.array([[Bx, By]]))[0] == 0: # 箭头指向非0侧
        Bx = np.percentile(plane_x, 25)
        By = (-1 / k) * (Bx - Ax) + Ay
    ax.quiver(Ax, Ay, (Bx-Ax), (By-Ay), angles='xy')
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel(model_name)
    ax.set_ylabel(model_name)
    
    ax.text(0.05, 0.025, "Precision: {:.4f}\nRecall: {:.4f}\nF1:{:.4f}".format(pre, rec, f1), 
            transform=plt.gca().transAxes, color='black', fontsize=20)
    
    # color bar
    labels = [str(i) for i in dates]
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     ticks=colors, fraction=0.05, pad=0.02, shrink=1.0)
    cb.ax.set_yticklabels(labels)
    plt.show()


def plot_verify_3d_movement_func(fos, vec_file_path, samples_num=100, views=(0, 0)):
    """ 
        观察单个主题在向量空间中移动 
        逐年散点3d图
    """
    global T0_, T2_
    model_name = vec_file_path.split('/')[-2]
    dic  = read_file(os.path.join(vec_file_path, fos)) # fos的向量字典
    beg_year = max(min(dic.keys()), T0_)               # fos的启始年
    end_year = min(max(dic.keys()), T2_)               # fos的末年
    vecs = get_vec_func(dic, beg_year, end_year)       # beg_year - end_year 所有向量
    
    # pca reduce dim
    pca = decomposition.PCA(n_components=3)            # pca 
    pca.fit(vecs)
    
    acc_nop = 0
    vecs_3d_ts = dict()
    for t in range(beg_year, end_year + 1):
        vecs_i = get_vec_func(dic, t, t)
        acc_nop += len(vecs_i)
        if acc_nop >= 10:
            if len(vecs_i) != 0:    
                vecs_3d_t = pca.transform(vecs_i)
                vecs_3d_ts[t] = vecs_3d_t
            else:
                vecs_3d_ts[t] = []
    ts = vecs_3d_ts.keys()
    beg_t  = min(ts)                                # fos的启始年
    end_t  = max(ts)                                # fos的最后一年 2018         
    dates  = np.arange(beg_t, end_t + 1)            # Time
    colors = np.linspace(-1, 1, len(dates))         # color bar
    middle_t = int((beg_t + end_t) / 2)

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 10
              }
    rcParams.update(config)
    ax = fig.add_subplot(111, projection='3d')
    # ax.grid(False)
    
    xyzcs = list()  # (x, y, z, color)
    svc_y = list()
    for i, t in enumerate(range(beg_t, end_t + 1)):
        vecs_3d_t = vecs_3d_ts[t]
        if len(vecs_3d_t) == 0:
            continue
        
        samples_size = min(len(vecs_3d_t), samples_num)
        print("{} : {}/{}".format(t, samples_size, len(vecs_3d_t)))
        vecs_3d_t = random.sample(list(vecs_3d_t), samples_size)
        vecs_3d_t = np.array(vecs_3d_t)
        xs = vecs_3d_t[:, 0]
        ys = vecs_3d_t[:, 1]
        zs = vecs_3d_t[:, 2]
        cs = colors[i] * np.ones(len(xs))
        xyzcs.append(np.array([xs, ys, zs, cs]))
        
        # (before middle_t 的是0类, after middle_t 是1类)
        if t > middle_t:
            svc_y += [1] * len(xs)
        else:
            svc_y += [0] * len(xs)
            
    cmap = 'seismic'
    xyzcs = np.concatenate(xyzcs, axis=-1)  
    ax.scatter(xyzcs[0, :], xyzcs[1, :], xyzcs[2, :], c=xyzcs[3, :], cmap=cmap, s=2, alpha=0.5)
    
    # 轴刻度
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_x   = math.ceil(max(xyzcs[0, :]) * 1000)
    down_x = math.floor(min(xyzcs[0, :]) * 1000)
    xticks = np.linspace(down_x, up_x, 5) / 1000
    
    up_y   = math.ceil(max(xyzcs[1, :]) * 1000)
    down_y = math.floor(min(xyzcs[1, :]) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    ax.set_yticks(yticks)
    
    up_z   = math.ceil(max(xyzcs[2, :]) * 1000)
    down_z = math.floor(min(xyzcs[2, :]) * 1000)
    zticks = np.linspace(down_z, up_z, 5) / 1000
    ax.set_zticks(zticks)
    plt.title(fos[:-4], fontsize=20, fontweight='bold')
    
    # plot hypeplane (before middle year 的是0类, after middle year 是1类)
    svc = svm.SVC(kernel='linear', C=1, class_weight='balanced')
    svc.fit(xyzcs[:3, :].T, svc_y)
    w = svc.coef_[0]
    k1 = - w[0] / w[2]
    k2 = - w[1] / w[2]
    b  = - svc.intercept_[0] / w[2]
    plane_x_, plane_y_ = np.meshgrid(np.linspace(min(xticks), max(xticks)),
                                     np.linspace(min(yticks), max(yticks)))
    plane_z_ = k1 * plane_x_ + k2 * plane_y_ + b
   
    # 只有 plane_bool 区域满足高度在zticks范围内
    plane_bool = list()
    plane_dist = list()
    for i in range(len(plane_x_)):
        plane_bool_i = list()
        plane_dist_i = list()
        for j in range(len(plane_x_)):
            z = plane_z_[i, j]
            if z >= zticks[0] and z <= zticks[-1]: 
                plane_bool_i.append(True)
            else:
                plane_bool_i.append(False)
            plane_dist_i.append(abs(z - (zticks[0] + zticks[1]) / 2))
        plane_bool.append(plane_bool_i)
        plane_dist.append(plane_dist_i)
    plane_bool = np.array(plane_bool)
    plane_dist = np.array(plane_dist)
    row_satisfied_points = np.sum(plane_bool, axis=1)
    avg_satisfied_points = np.sum(row_satisfied_points) / np.sum(row_satisfied_points > 0)
    row_satisfied = row_satisfied_points >= avg_satisfied_points
    col_satisfied = row_satisfied
    plane_x = plane_x_[row_satisfied][:, col_satisfied]
    plane_y = plane_y_[row_satisfied][:, col_satisfied]
    plane_z = k1 * plane_x + k2 * plane_y + b
    # plane_x, plane_y, plane_z = plane_x_, plane_y_, plane_z_
    surf = ax.plot_surface(plane_x, plane_y, plane_z, color='gray', alpha=0.5, label=r"Hyperplane: 0 ($\leq${}), 1 ($>${})".format(middle_t, middle_t))
    
    # 绘制法向量
    row_num, col_num = plane_x_.shape
    rsi = np.arange(len(row_satisfied))[row_satisfied]
    row_i = int(np.mean(rsi))
    col_i = row_i
    Ax = plane_x_[row_i][col_i]
    Ay = plane_y_[row_i][col_i]
    Az = k1 * Ax + k2 * Ay + b
    Bx = plane_x_[row_i][min(col_i+3, col_num-1)]
    By = k2 * (Bx - Ax) / k1 + Ay
    Bz = -1 * (Bx - Ax) / k1 + Az
    if svc.predict(np.array([[Bx, By, Bz]]))[0] == 0:
         Bx = plane_x_[row_i][max(col_i-3, 0)]
         By = k2 * (Bx - Ax) / k1 + Ay
         Bz = -1 * (Bx - Ax) / k1 + Az
    ABd = np.sqrt(((Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2)) * 10
    ax.quiver(Ax, Ay, Az, 
             (Bx-Ax)/ABd, (By-Ay)/ABd, (Bz-Az)/ABd, 
             pivot='middle', color="black")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    ax.set_zlim(zticks[0], zticks[-1])
    ax.set_xlabel(model_name)
    ax.set_ylabel(model_name)
    # 图视角
    ax.view_init(*views)  
    # 3d图的legend
    # surf._facecolors2d=surf._facecolors3d
    # surf._edgecolors2d=surf._edgecolors3d
    # ax.set_legend(frameon=False, loc='upper left')

    # color bar
    labels = [str(i) for i in dates]
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     ticks=colors, fraction=0.02, pad=0.025, shrink=1.0)
    cb.ax.set_yticklabels(labels)
    plt.show()


#%%
def plot_verify_2d_dissimilarity_func(fos_list, vec_file_path, model_name, samples_num=100):
    '''
    # 多个fos在空间内的相对位置关系
    Parameters
    ----------
    fos_list : list
        待可视化的主题.
    vec_file_path : string
        向量路径.
    samples_num : int, optional
        可视化采样数目. The default is 100.

    Returns
    -------
    None.

    '''
    global T0_, T2_
    
    all_vec_pool = dict()
    for fos in fos_list:
        dic  = read_file(os.path.join(vec_file_path, fos)) # fos的向量字典
        beg_year = max(min(dic.keys()), T0_)               # fos的启始年
        end_year = min(max(dic.keys()), T2_)               # fos的末年
        all_vec_pool[fos] = dict()
        acc_nop  = 0
        for t in range(beg_year, end_year + 1):
            vecs_t = get_vec_func(dic, t, t)
            acc_nop += len(vecs_t)
            if acc_nop >= 10:
                if len(vecs_t) != 0:
                    samples_size = min(len(vecs_t), samples_num)
                    print("{} {} : {}/{}".format(fos, t, samples_size, len(vecs_t)))
                    vecs_t_samples = random.sample(list(vecs_t), samples_size)
                    vecs_t_samples = np.array(vecs_t_samples)
                    
                    if model_name.lower() == "bert":
                       # topic embeddings generated by bert is sparse. To visualize, we adopt the normalization method in mpnet and MiniLM
                       # vecs_t_samples = np.array(vecs_t_samples, dtype=np.float32)
                       # vecs_t_samples = torch.nn.functional.normalize(torch.tensor(vecs_t_samples), p=2, dim=1).numpy()
                       # vecs_t_samples = np.array(vecs_t_samples, dtype=np.float16)
                       all_vec_pool[fos][t] = vecs_t_samples
                    else:
                       # topic embeddings generated by mpnet or MiniLM
                       all_vec_pool[fos][t] = vecs_t_samples
    # pca 训练
    all_vecs = list()
    for fos in all_vec_pool:
        for t in all_vec_pool[fos]:
            all_vecs.append(all_vec_pool[fos][t])
    all_vecs = np.concatenate(all_vecs, axis=0)
    
        
    pca = decomposition.PCA(n_components=2)            # pca 
    pca.fit(all_vecs)
    del all_vecs
    # pca 降维
    all_vec_pool_2d = dict()
    for fos in all_vec_pool:
        all_vec_pool_2d[fos] = dict()
        for t in all_vec_pool[fos]:
            vecs_t = all_vec_pool[fos][t]
            vecs_2d_t = pca.transform(vecs_t)
            all_vec_pool_2d[fos][t] = vecs_2d_t
    
    # 绘图
    colors = np.linspace(-1, 1, len(all_vec_pool_2d))            # color bar
    
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    ax = fig.add_subplot(111)
    
    xyzcs = list()  # (x, y, z, color)
    for i, fos in enumerate(all_vec_pool_2d):
        vecs_2d = list()
        svc_y   = list()
        ts = all_vec_pool_2d[fos].keys()
        if len(ts) == 0:
            print("{}在当前时间无数据".format(fos, 0))
            continue
        if len(ts) == 1:
            print("{}在当前时间仅有1年数据".format(fos)) # 只存在一年的数据, 故无法取分> middle_t 和< middle_t
            continue
        
        beg_t = min(ts)
        end_t = max(ts)
        middle_t = int((beg_t + end_t) / 2)
        for t in all_vec_pool_2d[fos]:
            vecs_2d.append(all_vec_pool_2d[fos][t])
            nop = len(all_vec_pool_2d[fos][t])
            if t > middle_t:
                svc_y += [1] * nop
            else:
                svc_y += [0] * nop
        vecs_2d = np.concatenate(vecs_2d, axis=0)
        xs = vecs_2d[:, 0]
        ys = vecs_2d[:, 1]
        cs = colors[i] * np.ones(len(xs))
        xyzcs.append(np.array([xs, ys, cs]))
    
        # 支持向量机分类 (确定类内移动方向)
        svc = svm.SVC(kernel='linear', C=1, class_weight='balanced')
        svc.fit(vecs_2d, svc_y)
        svc_y_pred = svc.predict(vecs_2d)
        # acc = sklearn.metrics.accuracy_score(svc_y, svc_y_pred)
        # sklearn.metrics.confusion_matrix(svc_y, svc_y_pred)
        # print(sklearn.metrics.classification_report(svc_y, svc_y_pred))
        svc_res = sklearn.metrics.classification_report(svc_y, svc_y_pred, output_dict=True, zero_division=0)
        pre = svc_res['macro avg']['precision']
        rec = svc_res['macro avg']['recall']
        f1  = svc_res['macro avg']['f1-score']
        pre0 = svc_res['0']['precision']
        rec0 = svc_res['0']['recall']
        f10  = svc_res['0']['f1-score']
          
        xticks = [min(xs), max(xs)]
        yticks = [min(ys), max(ys)]
        w = svc.coef_[0]
        k = - w[0] / w[1]
        b = - svc.intercept_[0] / w[1]
        plane_x_ = np.linspace(min(xticks), max(xticks))
        plane_x  = list()
        plane_y  = list()
        for x in plane_x_:
            y = k * x + b
            if y >= yticks[0] and y <= yticks[-1]:
                plane_x.append(x)
                plane_y.append(y)
                
        # 法向量的名字
        x1, x2 = np.mean(vecs_2d, axis=0)
        plt.text(x1, x2, fos[:-4], ha='left', wrap=True, fontsize=15)
                
        if len(plane_x) == 0 or f1 < 0.5:
            print("(未移动){}: {:.3f} / {:.3f} / {:.3f}".format(fos.ljust(40,"*"), pre, rec, f1))
            print("(未移动){}: {:.3f} / {:.3f} / {:.3f}".format("".ljust(40,"*"), pre0, rec0, f10))
            continue
        else:
            print("(已移动){}: {:.3f} / {:.3f} / {:.3f}".format(fos.ljust(40,"*"), pre, rec, f1))
            print("(已移动){}: {:.3f} / {:.3f} / {:.3f}".format("".ljust(40,"*"), pre0, rec0, f10))
        # ax.plot(plane_x, plane_y, c='black', linewidth=2, linestyle='--')
        # 法向量
        Ax = np.median(plane_x)
        Ay = k * Ax + b
        Bx = np.percentile(plane_x, 75)
        By = (-1 / k) * (Bx - Ax) + Ay
        if svc.predict(np.array([[Bx, By]]))[0] == 0: # 箭头指向非0侧
            Bx = np.percentile(plane_x, 25)
            By = (-1 / k) * (Bx - Ax) + Ay
            ax.quiver(Ax, Ay, (Bx-Ax), (By-Ay), angles='xy')
        else:
            ax.quiver(Ax, Ay, (Bx-Ax), (By-Ay), angles='xy')
        # # 法向量的名字
        # x1, x2 = np.mean(vecs_2d, axis=0)
        # plt.text(x1, x2, fos[:-4], ha='left', wrap=True, fontsize=15)
        
    cmap  = 'rainbow'
    xyzcs = np.concatenate(xyzcs, axis=-1)  
    ax.scatter(xyzcs[0, :], xyzcs[1, :], c=xyzcs[2, :], cmap=cmap, s=4, alpha=0.35)
    
    # 轴刻度
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_x   = math.ceil(max(xyzcs[0, :])  * 1000)
    down_x = math.floor(min(xyzcs[0, :]) * 1000)
    xticks = np.linspace(down_x, up_x, 5) / 1000
    ax.set_xticks(xticks)
    
    up_y   = math.ceil(max(xyzcs[1, :])  * 1000)
    down_y = math.floor(min(xyzcs[1, :]) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    ax.set_yticks(yticks)
    
    plt.title(model_name)
    # # color bar
    # labels = [str(i)[:-4] for i in fos_list]
    # norm = Normalize(vmin=colors.min(), vmax=colors.max())
    # cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
    #                   ticks=colors, fraction=0.2, pad=0.025, shrink=1.0)
    # cb.ax.set_yticklabels(labels)
    # plt.show()


def plot_verify_3d_dissimilarity_func(fos_list, vec_file_path, samples_num=100, views=(0, 0)):
    global T0_, T2_
    all_vec_pool = dict()
    for fos in fos_list:
        dic   = read_file(os.path.join(vec_file_path, fos)) # fos的向量字典 
        beg_t = min(dic.keys())                             # fos的启始年
        end_t = max(dic.keys())                             # fos的末年
        all_vec_pool[fos] = dict()
        acc_nop = 0
        for t in range(beg_t, end_t + 1):
            vecs_t = get_vec_func(dic, t, t)
            acc_nop += len(vecs_t)
            if acc_nop >= 10:
                if len(vecs_t) != 0:
                    samples_size = min(len(vecs_t), samples_num)
                    # print("{} {} : {}/{}".format(fos, t, samples_size, len(vecs_t)))
                    vecs_t_samples = random.sample(list(vecs_t), samples_size)
                    vecs_t_samples = np.array(vecs_t_samples)
                    all_vec_pool[fos][t] = vecs_t_samples
    # pca 训练
    all_vecs = list()
    for fos in all_vec_pool:
        for year in all_vec_pool[fos]:
            all_vecs.append(all_vec_pool[fos][year])
    all_vecs = np.concatenate(all_vecs, axis=0)
    pca = decomposition.PCA(n_components=3)            # pca 
    pca.fit(all_vecs)
    del all_vecs
    # pca 降维
    all_vec_pool_3d = dict()
    for fos in all_vec_pool:
        all_vec_pool_3d[fos] = dict()
        for t in all_vec_pool[fos]:
            vecs_3d = pca.transform(all_vec_pool[fos][t])
            all_vec_pool_3d[fos][t] = vecs_3d
    
    # 绘图
    colors = np.linspace(-1, 1, len(all_vec_pool_3d))            # color bar
    
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 10
              }
    rcParams.update(config)
    ax = fig.add_subplot(111, projection='3d')
    
    xyzcs = list()  # (x, y, z, color)
    for i, fos in enumerate(all_vec_pool_3d):
        vecs_3d = list()
        svc_y = list()
        ts    = all_vec_pool_3d[fos].keys()
        beg_t = min(ts)
        end_t = max(ts)
        middle_t = int((beg_t + end_t) / 2)
        for t in all_vec_pool_3d[fos]:
            vecs_3d.append(all_vec_pool_3d[fos][t])
            nop = len(all_vec_pool_3d[fos][t])
            if t > middle_t:
                svc_y += [1] * nop
            else:
                svc_y += [0] * nop
        vecs_3d = np.concatenate(vecs_3d, axis=0)
        
        xs = vecs_3d[:, 0]
        ys = vecs_3d[:, 1]
        zs = vecs_3d[:, 2]
        cs = colors[i] * np.ones(len(xs))
        xyzcs.append(np.array([xs, ys, zs, cs]))
        
        # 支持向量机分类 (确定类内移动方向)
        svc = svm.SVC(kernel='linear', C=1, class_weight='balanced')
        svc.fit(vecs_3d, svc_y)
        xticks = [min(xs), max(xs)]
        yticks = [min(ys), max(ys)]
        zticks = [min(zs), max(zs)]
        w = svc.coef_[0]
        k1 = - w[0] / w[2]
        k2 = - w[1] / w[2]
        b  = - svc.intercept_[0] / w[2]
        plane_x_, plane_y_ = np.meshgrid(np.linspace(min(xticks), max(xticks)),
                                         np.linspace(min(yticks), max(yticks)))
        plane_z_ = k1 * plane_x_ + k2 * plane_y_ + b
        # 只有 plane_bool 区域满足高度在zticks范围内
        plane_bool = list()
        plane_dist = list()
        for i in range(len(plane_x_)):
            plane_bool_i = list()
            plane_dist_i = list()
            for j in range(len(plane_x_)):
                z = plane_z_[i, j]
                if z >= zticks[0] and z <= zticks[-1]: 
                    plane_bool_i.append(True)
                else:
                    plane_bool_i.append(False)
                plane_dist_i.append(abs(z - (zticks[0]+zticks[1])/2))
            plane_bool.append(plane_bool_i)
            plane_dist.append(plane_dist_i)
        plane_bool = np.array(plane_bool)
        plane_dist = np.array(plane_dist)
        
        row_satisfied_points = np.sum(plane_bool, axis=1)
        if np.sum(row_satisfied_points > 0) == 0:
            continue
        avg_satisfied_points = np.sum(row_satisfied_points) / np.sum(row_satisfied_points > 0)
        row_satisfied = row_satisfied_points >= avg_satisfied_points
        col_satisfied = row_satisfied
        plane_x = plane_x_[row_satisfied][:, col_satisfied]
        plane_y = plane_y_[row_satisfied][:, col_satisfied]
        plane_z = k1 * plane_x + k2 * plane_y + b
        # ax.plot_surface(plane_x, plane_y, plane_z, color='gray', alpha=0.5)
        
        # 绘制法向量
        row_num, col_num = plane_x_.shape
        rsi = np.arange(len(row_satisfied))[row_satisfied]
        row_i = int(np.mean(rsi))
        col_i = row_i
        Ax = plane_x_[row_i][col_i]
        Ay = plane_y_[row_i][col_i]
        Az = k1 * Ax + k2 * Ay + b
        Bx = plane_x_[row_i][min(col_i+3, col_num-1)]
        By = k2 * (Bx - Ax) / k1 + Ay
        Bz = -1 * (Bx - Ax) / k1 + Az
        if svc.predict(np.array([[Bx, By, Bz]]))[0] == 0:
            Bx = plane_x_[row_i][max(col_i-3, 0)]
            By = k2 * (Bx - Ax) / k1 + Ay
            Bz = -1 * (Bx - Ax) / k1 + Az
             
        ABd = np.sqrt(((Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2)) * 10
        ax.quiver(Ax, Ay, Az, 
                  (Bx-Ax)/ABd, (By-Ay)/ABd, (Bz-Az)/ABd, 
                  pivot='middle', color="black")
            
    cmap  = 'rainbow'
    xyzcs = np.concatenate(xyzcs, axis=-1)  
    ax.scatter(xyzcs[0, :], xyzcs[1, :], xyzcs[2, :], c=xyzcs[3, :], cmap=cmap, s=2, alpha=0.35)

    # 轴刻度
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_x   = math.ceil(max(xyzcs[0, :]) * 1000)
    down_x = math.floor(min(xyzcs[0, :]) * 1000)
    xticks = np.linspace(down_x, up_x, 5) / 1000
    
    up_y   = math.ceil(max(xyzcs[1, :]) * 1000)
    down_y = math.floor(min(xyzcs[1, :]) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    ax.set_yticks(yticks)
    
    up_z   = math.ceil(max(xyzcs[2, :]) * 1000)
    down_z = math.floor(min(xyzcs[2, :]) * 1000)
    zticks = np.linspace(down_z, up_z, 5) / 1000
    ax.set_zticks(zticks)
    # plt.title(fos[:-4], fontsize=20, fontweight='bold')
    
    ax.view_init(*views)  # 图视角
    # color bar
    labels = [str(i)[:-4] for i in fos_list]
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=ax,
                     ticks=colors, fraction=0.02, pad=0.025, shrink=1.0)
    cb.ax.set_yticklabels(labels)
    plt.show()


#%%
def plot_yearly_new_topics(fos_mu_std_nop, fos_L3_BY):
    # fos_mu_std_nop 只包含累计频率超过100的fos_L3, 它们被用作训练decoder
    # filtered_nop = 500 进一步筛选主题词
    # 绘制逐年新生的主题词数目
    
    def filter_fos_by_freq(fos_mu_std_nop, filtered_nop, fos_L3_BY):
        fos_mu_std_nop2 = dict()
        for fos in fos_mu_std_nop:
            if len(fos_mu_std_nop[fos].keys()) == 0:
                continue
            end_t = max(fos_mu_std_nop[fos].keys())
            acc_nop_t = fos_mu_std_nop[fos][end_t][-1]
            if acc_nop_t >= filtered_nop:
                fos_mu_std_nop2[fos] = fos_mu_std_nop[fos]
        # 绘制逐年新主题数
        t2n = dict() # Key: time Value: topics
        for fos in fos_mu_std_nop2:
            t = fos_L3_BY[fos]
            if t not in t2n:
                t2n[t] = list()
            t2n[t].append(fos)
        Y = dict()   # Key: time Value: number of topics
        cut_t = 1900 # 1990年之前
        for t in t2n:
            if t <= cut_t:
                if cut_t not in Y:
                    Y[cut_t] = 0
                Y[cut_t] += len(t2n[t])
            else:
                if t not in Y:
                    Y[t] = len(t2n[t])
        X = sorted(list(Y.keys()))
        return X, Y, fos_mu_std_nop2
    
    X2, Y2, fos_mu_std_nop2 = filter_fos_by_freq(fos_mu_std_nop, 500, fos_L3_BY)
    X3, Y3, fos_mu_std_nop3 = filter_fos_by_freq(fos_mu_std_nop, 100, fos_L3_BY)

    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 25
              }
    rcParams.update(config)
    
    plt.plot(X2, [Y2[t] for t in X2], c='red',  linestyle='-',  marker="*", linewidth=1.5, markersize=7, label=r"$FoS_{L2}$($\geq500$)") 
    plt.plot(X3, [Y3[t] for t in X3], c='blue', linestyle='--', marker="+", linewidth=1.5, markersize=7, label=r"$FoS_{L2}$($\geq100$)")
    plt.text(0.55, 0.15, r"{}".format(len(fos_mu_std_nop2)) + r' $FoS_{L2}$', transform=plt.gca().transAxes, color='red', fontsize=20)
    plt.text(0.67, 0.50, r"{}".format(len(fos_mu_std_nop3)) + r' $FoS_{L2}$', transform=plt.gca().transAxes, color='blue', fontsize=20)
    plt.xticks(np.arange(1900, 2021, 10), rotation=45)
    plt.yticks(np.arange(0, 300, 50))
    plt.ylabel("Number of new topics", fontsize=25)
    plt.xlabel("Time", fontsize=25)
    plt.legend(frameon=False)
    return fos_mu_std_nop2, fos_mu_std_nop3


def plot_movement_distribution_basedon_ED(fos_mu_std_nop2, fos_mu_std_nop3, model_name):
    # 统计所有主题在T0_至于T2_年的质心移动情况 Euclidean Distance
    def calculate_Euclidean_distance(fos_mu_std_nop2):
        dist_dis2 = dict()
        for fos in fos_mu_std_nop2:
            ts = list(fos_mu_std_nop2[fos].keys())
            ts_= list()
            for t in ts:
                if t >= T0_ and t <= T2_:
                    ts_.append(t)
            if len(ts_) <= 1:
                continue
            beg_t = min(ts_)
            end_t = max(ts_)
            centroid_beg = fos_mu_std_nop2[fos][beg_t][0]
            centroid_end = fos_mu_std_nop2[fos][end_t][0]
            dist = np.sqrt(np.sum(np.square(centroid_beg-centroid_end))) 
            dist_dis2[fos] = (dist)
        return dist_dis2        
    
    dist_dis2 = calculate_Euclidean_distance(fos_mu_std_nop2)
    dist_dis3 = calculate_Euclidean_distance(fos_mu_std_nop3)
    
    dist_dis2_ = [dist_dis2[fos] for fos in dist_dis2]
    dist_dis3_ = [dist_dis3[fos] for fos in dist_dis3]
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    sns.distplot(dist_dis2_, hist=True, kde=True, rug=False,
                 bins=30,
                 hist_kws = {'rwidth':1, 'color':'red', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.5, "label": r"$FoS_{L2}$($\geq500$)"},
                 kde_kws  = {"color": "black", "alpha":0.5, "linewidth": 2, "shade":False, "label": ""},
                 rug_kws  = {"color": "black",  "alpha":0.25, "linewidth": 0.01, "height":0.05},
                 fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""}
                 )
    sns.distplot(dist_dis3_, hist=True, kde=True, rug=False,
                 bins=30,
                 hist_kws = {'rwidth':1, 'color':'darkblue', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.5, "label": r"$FoS_{L2}$($\geq100$)"},
                 kde_kws  = {"color": "black", "alpha":0.5, "linewidth": 2,"shade":False, "label": "", "linestyle":"-"},
                 rug_kws  = {"color": "black",  "alpha":0.25, "linewidth": 0.01, "height":0.05},
                 fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""}
                 )
    plt.xlabel(r"Euclidean distance")
    # plt.xlabel(r"$\Vert{\overline{X_i(2010)}-\overline{X_i(1990)}}\Vert_2$")
    plt.ylabel(r"Density")
    if model_name == "bert": model_name = "bert-base-uncased"
    plt.title(model_name)
    # bert
    plt.xticks(np.arange(0, 6, 1))
    plt.xlim(0, 5)
    plt.yticks(np.arange(0, 1.6, 0.3))
    # # mppnet
    # plt.xticks(np.arange(0, 0.65, 0.15))
    # plt.xlim(0, 0.6)
    # plt.yticks(np.arange(0, 10, 2))
    plt.legend(frameon=False)
    
    return dist_dis2, dist_dis3


def plot_movement_distribution_basedon_SVM(fos_mu_std_nop2, fos_mu_std_nop3, model_name):
    
    global T0_, T2_
    
    def conduct_svm_classifier():
        fos2SVM = dict()
        samples_num = 100
        for fos in tqdm(fos_mu_std_nop3):
            dic  = read_file(os.path.join(vec_file_path, fos + ".pkl")) # fos的向量字典
            beg_year = max(min(dic.keys()), T0_)               # fos的启始年
            end_year = min(max(dic.keys()), T2_)               # fos的末年
            vecs = get_vec_func(dic, beg_year, end_year)       # beg_year - end_year 所有向量
            if len(vecs) <= 1:
                continue
            # pca reduce dim
            pca = decomposition.PCA(n_components=2)            # pca 
            pca.fit(vecs)
            
            acc_nop = 0
            vecs_2d_ts = dict()
            for t in range(beg_year, end_year + 1):
                if t > T2_ or t < T0_:
                    continue
                vecs_t = get_vec_func(dic, t, t)
                acc_nop += len(vecs_t)
                if acc_nop >= 10:
                    if len(vecs_t) != 0:    
                        vecs_2d_t = pca.transform(vecs_t)
                        vecs_2d_ts[t] = vecs_2d_t
                    else:
                        vecs_2d_ts[t] = []   
            if len(vecs_2d_ts) <= 1:
                continue
            ts     = vecs_2d_ts.keys()
            beg_t  = min(ts)                               # fos的启始年
            end_t  = max(ts)                               # fos的最后一年 2018         
            dates  = np.arange(beg_t, end_t + 1)           # Time
            colors = np.linspace(-1, 1, len(dates))        # color bar 
            middle_t = int((beg_t + end_t) / 2)
            
            svc_y = list()
            xyzcs = list()
            for i, t in enumerate(range(beg_t, end_t + 1)):
                vecs_2d_t = vecs_2d_ts[t]
                if len(vecs_2d_t) == 0:
                    continue
                samples_size = min(len(vecs_2d_t), samples_num)
                # print("{} : {}/{}".format(t, samples_size, len(vecs_2d_t)))
                vecs_2d_t = random.sample(list(vecs_2d_t), samples_size)
                vecs_2d_t = np.array(vecs_2d_t)
                xs = vecs_2d_t[:, 0]
                ys = vecs_2d_t[:, 1]
                cs = colors[i] * np.ones(len(xs))
                xyzcs.append(np.array([xs, ys, cs]))
                # (before middle_t 的是0类, after middle_t 是1类)
                if t > middle_t:
                    svc_y += [1] * len(xs)
                else:
                    svc_y += [0] * len(xs)
            if len(set(svc_y)) <= 1:
                continue
            xyzcs = np.concatenate(xyzcs, axis=-1) 
            # plot hypeplane (before middle year 的是0类, after middle year 是1类)
            svc = svm.SVC(kernel='linear', C=1, class_weight='balanced')
            svc.fit(xyzcs[:2, :].T, svc_y)
            svc_y_pred = svc.predict(xyzcs[:2, :].T)
            # 准确率, 召回率, 宏f1
            svc_res = sklearn.metrics.classification_report(svc_y, svc_y_pred, output_dict=True, zero_division=0)
            pre = svc_res['macro avg']['precision']
            rec = svc_res['macro avg']['recall']
            f1  = svc_res['macro avg']['f1-score']
            fos2SVM[fos] = (pre, rec, f1)
        save_file(fos2SVM, "./Verify/fos2SVM({}).pkl".format(model_name))
    
    tmp_path = "./Verify/fos2SVM({}).pkl".format(model_name)
    if os.path.exists(tmp_path):
        fos2SVM = read_file(tmp_path)
    else:
        conduct_svm_classifier()
        fos2SVM = read_file(tmp_path)
        
    X3 = [fos2SVM[fos][-1] for fos in fos2SVM]
    X2 = list()
    for fos in fos2SVM:
        if fos in fos_mu_std_nop2:
            X2.append(fos2SVM[fos][-1])
        
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    kwargs = {'cumulative': True}
    sns.distplot(X2, bins=100, hist_kws=kwargs, kde_kws=kwargs, hist=False, label=r"$FoS_{L2}$($\geq500$)", color='red')
    kwargs = {'cumulative': True, 'linestyle': '--'}
    sns.distplot(X3, bins=100, hist_kws=kwargs, kde_kws=kwargs, hist=False, label=r"$FoS_{L2}$($\geq100$)", color='blue')
    plt.plot([0.5, 0.5], [0, 1], c='black', linestyle='--')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlim(0, 1)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.ylim(0, 1)
    if model_name == "bert": model_name = "bert-base-uncased"
    plt.title(model_name)
    plt.ylabel("CDF")
    plt.xlabel('F1 score')
    plt.legend(frameon=False)
    
    
def plot_cor_distance_and_cooccuredfos(fos_mu_std_nop2, fos_mu_std_nop3, dist_dis2, dist_dis3, model_name, filterb):
    # (T0_至T2_质心的距离) 与 (截止T2_时期, 与fos_i共现的fos_j的数目) 的关系
    # 上述两者呈现负相关性, 即与更少主题共现的主题反而移动距离更大,
    # 可能是更受某个主题影响, 朝着它移动
    
    def lienar_func(x, a, b):
        y = np.multiply(x, a) + b
        return y
    
    def calculate_number_of_cooccurred(fos_mu_std_nop2, comatrix, filterb=False):
        inc_num_dis2 = dict()
        for fos in fos_mu_std_nop2:
            ts = list(comatrix[fos].keys())
            ts_= list()
            for t in ts:
                if t >= T0_ and t <= T2_:
                    ts_.append(t)
            if len(ts_) <= 1:
                continue
            # 待分析时间段[beg_t, end_t]
            beg_t = min(ts_)
            end_t = max(ts_)
            # 截止beg_t和end_t时刻, 与fos共现的fos_j集合
            beg_t_set = dict()
            end_t_set = dict()
            for t in comatrix[fos]:
                if t <= beg_t:
                    for fos_j in comatrix[fos][t]:
                        if fos_j not in fos_mu_std_nop2:
                            continue 
                        if fos_j not in beg_t_set:
                            beg_t_set[fos_j]  = comatrix[fos][t][fos_j]
                        else:
                            beg_t_set[fos_j] += comatrix[fos][t][fos_j]
                if t <= end_t:
                    for fos_j in comatrix[fos][t]:
                        if fos_j not in fos_mu_std_nop2:
                            continue
                        if fos_j not in end_t_set:
                            end_t_set[fos_j]  = comatrix[fos][t][fos_j]
                        else:
                            end_t_set[fos_j] += comatrix[fos][t][fos_j]            
            # 统计差异
            if filterb:
                # 过滤低频率共现fos_j by median of frequency
                inc_num = 0
                if len(beg_t_set) > 1:
                    # 超过median frequency的主题频次
                    cut = np.median([beg_t_set[fos_j] for fos_j in beg_t_set])
                    for fos_j in beg_t_set:
                        if beg_t_set[fos_j] > cut:
                            inc_num += 1
                inc_num_dis2[fos] = inc_num
            else:
                # 不过低频率贡献fos_j
                inc_num_dis2[fos] = len(end_t_set)
        return inc_num_dis2
    
    comatrix = read_file(os.path.join(data_file_path, "comatrix"))
    
    inc_num_dis2 = calculate_number_of_cooccurred(fos_mu_std_nop2, comatrix, filterb)
    inc_num_dis3 = calculate_number_of_cooccurred(fos_mu_std_nop3, comatrix, filterb)
    
    X1, X2 = list(), list()
    for fos in dist_dis2:
        X1.append(dist_dis2[fos])
        X2.append(inc_num_dis2[fos])
    X1, X2 = np.array(X1), np.array(X2)
    X3, X4 = list(), list()
    for fos in dist_dis3:
        X3.append(dist_dis3[fos])
        X4.append(inc_num_dis3[fos])
    X3, X4 = np.array(X3), np.array(X4)

    popt12, pcov = curve_fit(lienar_func, np.log(np.maximum(X2, 1e-3)), X1)
    X1_pred = lienar_func(np.log(np.arange(1, 1000, 10)), *popt12)
    cor12 = np.corrcoef(X1, X2)[0, 1]
    
    popt34, pcov = curve_fit(lienar_func, np.log(np.maximum(X4, 1e-3)), X3)
    X4_pred = lienar_func(np.log(np.arange(1, 1000, 10)), *popt34)
    cor34 = np.corrcoef(X3, X4)[0, 1]
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    plt.scatter(X2, X1, marker='*', s=1, c='red')
    plt.plot(np.arange(1, 1000, 10), X1_pred, c='red', linewidth=3, linestyle='-', label=r'$FoS_{L2}$($\geq500$)'+':{:.4f}x+{:.4f}'.format(*popt12))
    
    plt.scatter(X4, X3, marker='+', s=1, c='blue')
    plt.plot(np.arange(1, 1000, 10), X4_pred, c='blue', linewidth=3, linestyle='--', label=r'$FoS_{L2}$($\geq100$)'+':{:.4f}x+{:.4f}'.format(*popt34))
    
    plt.text(0.05, 0.2, "Pearsonr:{:.4f}".format(cor12), # 0.6, 0.6  # 0.05, 0.3 # 0.05, 0.6
            transform=plt.gca().transAxes, color='red', fontsize=20)
    plt.text(0.05, 0.1, "Pearsonr:{:.4f}".format(cor34), # 0.6, 0.5  # 0.05, 0.2 # 0.05, 0.5
            transform=plt.gca().transAxes, color='blue', fontsize=20)
    if model_name == "bert": model_name = "bert-base-uncased"
    plt.title(model_name)
    plt.xscale('log')
    plt.ylim(0, .5, 1)
    plt.ylabel(r"Euclidean distance")
    plt.xlabel(r"Number of topics studied with topic $i$" )
    plt.legend(frameon=False)


#%% 
if __name__ == "__main__":
    # Section 3.2 Demonstrating the motion of topic emebeddings
    # 选取一个时间段进行移动可视化
    T0_ = 1990
    T2_ = 2010
    
    def main():
        # model_name = "all-MiniLM-L6-v2"
        model_name = "all-mpnet-base-v2"
        # model_name = "bert"
        save_file_path = os.path.join(data_file_path, model_name)
        vec_file_path  = os.path.join(save_file_path, "FoS2Vec")
       
        fos_L3_BY = read_file(os.path.join(data_file_path, "fos_L3_BY.pkl"))
        filtered_fos_for_decoder = read_file(os.path.join(data_file_path, "filtered_fos_for_decoder.pkl")) # 此处筛选>=100, 第一篇论文是>100
        # 计算fos的累计逐年均值和标准差
        bAccumulative = False
        fos_mu_std_file_path = os.path.join(save_file_path, "fos_mu_std_nop({}).pkl".format(bAccumulative))
        if os.path.exists(fos_mu_std_file_path):
            fos_mu_std_nop = read_file(fos_mu_std_file_path)
        else:
            fos_mu_std_nop = calculate_accumulative_mu_std_MP(filtered_fos_for_decoder, vec_file_path, bAccumulative)
            save_file(fos_mu_std_nop,fos_mu_std_file_path)
        
        # (0) 观察逐年的新生主题数目
        fos_mu_std_nop2, fos_mu_std_nop3 = plot_yearly_new_topics(fos_mu_std_nop, fos_L3_BY)
        
        # (0) 统计全部主题的移动距离分布 (Euclidean distance)
        dist_dis2, dist_dis3 = plot_movement_distribution_basedon_ED(fos_mu_std_nop2, fos_mu_std_nop3, model_name) # 
        
        # (0) SVM 检验运动
        plot_movement_distribution_basedon_SVM(fos_mu_std_nop2, fos_mu_std_nop3, model_name)
        
        # (0) 主题的移动距离 和 新增合作fos数目的关系
        plot_cor_distance_and_cooccuredfos(fos_mu_std_nop2, fos_mu_std_nop3, dist_dis2, dist_dis3, model_name, False)
        plot_cor_distance_and_cooccuredfos(fos_mu_std_nop2, fos_mu_std_nop3, dist_dis2, dist_dis3, model_name, True)
        
        # (1) 投影向量可以观察到明确的方向变化
        fos = 'body of knowledge' 
        # ontology # "information processing" 
        # image stitching # paging  # network interface # digitization
        # preprocessor # jamming # electronic form  # semantic similarity
        fos = random.sample(fos_mu_std_nop2.keys(), 1)[0] 
        plot_verify_1d_movement_func(fos, fos_mu_std_nop, model_name, dim=-1)
        plot_verify_2d_movement_func(fos + ".pkl", vec_file_path, 100)
        plot_verify_3d_movement_func(fos + ".pkl", vec_file_path, 100, (10, 80))
        
        # (2) 不同fos的投影向量之间存在差距
        fos_list = [
            'linear combination', 
            'digital signal processing', 
            'simulation system',
            'decision analysis', 
            'code reading', 
            'vehicle dynamics',
            'sql'
        ] # 'metacomputing'
        # fos_list = random.sample(fos_mu_std_nop.keys(), 10)
        fos_list = [fos + ".pkl" for fos in fos_list]
        plot_verify_2d_dissimilarity_func(fos_list, vec_file_path, model_name, 100)
        plot_verify_3d_dissimilarity_func(fos_list, vec_file_path, 100, (10, 100))
    
    
    
