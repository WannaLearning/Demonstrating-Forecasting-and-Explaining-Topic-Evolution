#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:04:51 2023

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
import multiprocessing
import pandas as pd
import prettytable as pt
import sklearn
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from tqdm import tqdm
from sklearn import decomposition
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

from Semantic_Movement_Predict import *
from Extract_cs_data import *
from vec_by_bertvae import *
from vec_by_mpnet import *
from vec_by_bert import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def sampling_from_gaussian(mu, std, size):
    vecs = np.random.multivariate_normal(mu, std, size)
    return vecs


def sampling_from_pred_direction(pred_vec, yearly_sampling_size, method):
    ''' 预测向量 '''
    new_vecs = dict()
    if method == 'mean':
        # 每个fos在每年t的mu预测作为预测向量 ~ mu(t)
        for fos in tqdm(pred_vec):
            if len(pred_vec[fos]) == 1:
                new_vecs[fos] = pred_vec[fos]
            else:
                new_vecs[fos] = pred_vec[fos][0].T  # idx==0 表示只取mu(t) # 维度是: Time x Embedding_dim
    elif method == 'sampling':
        # 每个fos在每年t的mu和std中采样作为预测向量 ~ sampling from Normal(mu(t), std(t))
        total_sampling_size = 0
        for fos in pred_vec:
            mu_pred_ts, std_pred_ts = pred_vec[fos]
            # 确定统一采样大小
            dim_len, time_len    = mu_pred_ts.shape
            fos_sampling_size    = yearly_sampling_size * time_len
            total_sampling_size += fos_sampling_size  
        mu_pred_t  = mu_pred_ts[:,  -1]
        std_pred_t = std_pred_ts[:, -1]
        # 统一采样 (从多元标准化正态分布)
        samples = sampling_from_gaussian(np.zeros(mu_pred_t.shape), np.diag(np.ones(std_pred_t.shape)), total_sampling_size)
        samples = np.array(samples, dtype=np.float16)
    
        # 统一变换
        mu_list  = list()
        std_list = list()
        for fos in tqdm(pred_vec):
            mu_pred_ts, std_pred_ts = pred_vec[fos]
            dim_len, time_len = mu_pred_ts.shape
            # 每年的均值和标准差中采样*个向量
            for t in range(time_len):
                mu_pred_t  = mu_pred_ts[:,  t]
                std_pred_t = std_pred_ts[:, t]
                std_pred_t = np.maximum(std_pred_t, 1e-5)
                std_pred_t = np.ones(std_pred_t.shape) * 1e-2 # *** 限制方差 ***
                for k in range(yearly_sampling_size):
                    mu_list.append([mu_pred_t])
                    std_list.append([std_pred_t])
        mu_list  = np.concatenate(mu_list)
        std_list = np.concatenate(std_list)
        samples  = mu_list + np.multiply(samples, std_list)
        # 
        start_idx = 0
        end_idx   = 0
        for fos in pred_vec:
            mu_pred_ts, std_pred_ts = pred_vec[fos]
            dim_len, time_len = mu_pred_ts.shape
            end_idx += time_len * yearly_sampling_size
            new_vecs[fos] = samples[start_idx: end_idx, :]   # 维度是: Time x Embedding_dim
            start_idx = end_idx
    else:
        pass
    return new_vecs


def sampling_from_pred_direction_MP(yearly_sampling_size, method, 
                                    pred_model, model_name,
                                    loop_num=3, mp_num=6):
    """ 预测向量 """
    # 模型直接预测的向量
    pred_vec  = read_file(os.path.join(res_explain_path, "pred_vec(var)({})".format(model_name)))
    # 若mean方法, 则直接采用pred_vec = pred_vec2; 若sampling方法, 则需要从正态分布中采样
    pred_vec2 = dict()
    
    fos_list   = list(pred_vec.keys())
    total_size = len(fos_list)
    loop_size  = math.ceil(total_size / loop_num)
    
    count = 0 
    start_l_idx = 0
    end_l_idx   = 0
    for l in range(loop_num):
        end_l_idx   = min(start_l_idx + loop_size, total_size)
        fos_list_l  =  fos_list[start_l_idx: end_l_idx]
        start_l_idx = end_l_idx
    
        # 创建进程池 - l
        start_idx = 0
        end_idx   = 0
        total_l_size = len(fos_list_l)
        batch_size = math.ceil(total_l_size/ mp_num)
        results = list()
        pool = multiprocessing.Pool(processes=mp_num)      
        for mp_i in range(mp_num):
            end_idx    = min(start_idx + batch_size, total_l_size)
            pred_vec_i = {fos: pred_vec[fos] for fos in fos_list_l[start_idx: end_idx]} 
            start_idx  = end_idx
            results.append(pool.apply_async(sampling_from_pred_direction, (pred_vec_i, yearly_sampling_size, method, )))
        pool.close()
        pool.join()
        
        for res in results:
            count += 1
            res_i = res.get()
            pred_vec2_i = dict()
            for fos in res_i:
                pred_vec2_i[fos] = res_i[fos]
            save_file(pred_vec2_i, os.path.join(res_explain_path, "pred_vec_exp({})({})({})".format(pred_model, model_name, count)))
                      
    del pred_vec
    
    # 合并
    for i in range(1, count+1):
        pred_vec2_i = read_file(os.path.join(res_explain_path, "pred_vec_exp({})({})({})".format(pred_model, model_name, i)))
        for fos in pred_vec2_i:
            pred_vec2[fos] = pred_vec2_i[fos]
    save_file(pred_vec2, os.path.join(res_explain_path, "pred_vec_exp({})({})({})".format(method, pred_model, model_name)))
    
    # 删除
    for i in range(1, count+1):
        del_path = os.path.join(res_explain_path, "pred_vec_exp({})({})({})".format(pred_model, model_name, i))
        if os.path.exists(del_path):
            os.remove(del_path)

#%%
def sampling_from_random_normal(pred_vec, yearly_sampling_size):
    # 从随机正态分布中采样
    total_sampling_size = 0
    new_vecs = dict()
    for fos in tqdm(pred_vec):
        mu_ts = pred_vec[fos]
        time_len, dim_len = mu_ts.shape
        sampling_size = time_len * yearly_sampling_size
        total_sampling_size += sampling_size
    samples = sampling_from_gaussian(np.zeros(dim_len), np.diag(np.ones(dim_len)), total_sampling_size)
    samples = np.array(samples, dtype=np.float16)
    
    # 和mpnet进行同样的normalizer
    samples = F.normalize(torch.tensor(samples, dtype=torch.float32), p=2, dim=1).numpy()
    start_idx = 0
    end_idx   = 0
    for fos in pred_vec:
        mu_ts = pred_vec[fos]
        time_len, dim_len = mu_ts.shape
        end_idx += time_len * yearly_sampling_size
        new_vecs[fos] = np.array(samples[start_idx: end_idx, :], dtype=np.float16)
        start_idx = end_idx
    return new_vecs


def sampling_from_random_normal_MP(pred_vec, yearly_sampling_size, mp_num=8):
    fos_list  = list(pred_vec.keys())
    total_num = len(fos_list)
    batch_num = math.ceil(total_num/ mp_num)
    start_idx = 0
    end_idx   = 0
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        end_idx    = min(start_idx + batch_num, total_num)
        fos_list_i = fos_list[start_idx: end_idx]
        pred_vec_i = {fos: pred_vec[fos] for fos in fos_list_i} 
        start_idx  = end_idx
        results.append(pool.apply_async(sampling_from_random_normal, (pred_vec_i, yearly_sampling_size, )))
    pool.close()
    pool.join()
    
    new_vecs = dict()
    for res in results:
        new_vecs_i = res.get()
        for fos in new_vecs_i:
            new_vecs[fos] = new_vecs_i[fos]
            

#%%
def sampling_from_train_direction(train_input, train_output, yearly_sampling_size, method):
    ''' 从训练集得到的方向计算预测向量 '''
    new_vecs = dict()
    if method == 'mean':
        # 每个fos在每年t的mu预测作为预测向量 ~ mu(t)
        for fos in tqdm(train_input):
            fos_sampling_vecs = list()
            for year in train_input[fos]:
                mu_t, std_t, _ = train_input[fos][year]
                fos_sampling_vecs.append([mu_t])
            for year in train_output[fos]:
                mu_t, std_t, _ = train_output[fos][year]
                fos_sampling_vecs.append([mu_t])
            fos_sampling_vecs = np.concatenate(fos_sampling_vecs)
            new_vecs[fos] = fos_sampling_vecs
    elif method == 'sampling':
        # 每个fos在每年t的mu和std中采样作为预测向量 ~ sampling from Normal(mu(t), std(t))
        # 统一采样
        total_sampling_size = 0
        for fos in tqdm(train_input):
            time_len = len(train_input[fos]) + len(train_output[fos])
            fos_sampling_size    = yearly_sampling_size * time_len
            total_sampling_size += fos_sampling_size  
        for year in train_input[fos]:
            mu_t, std_t, _  = train_input[fos][year]
            break
        samples = sampling_from_gaussian(np.zeros(mu_t.shape), 
                                         np.diag(np.ones(std_t.shape)), total_sampling_size)
        samples = np.array(samples, dtype=np.float16)
        
        # 统一变换
        mu_list  = list()
        std_list = list()
        for fos in tqdm(train_input):
            for year in train_input[fos]:
                mu_t, std_t, _ = train_input[fos][year]
                std_t = np.maximum(std_t, 1e-5)
                std_t = np.ones(std_t.shape) * 1e-2 # *** 限制方差 ***
                for k in range(yearly_sampling_size):
                    mu_list.append([mu_t])
                    std_list.append([std_t])
            for year in train_output[fos]:
                mu_t, std_t, _ = train_output[fos][year]
                std_t = np.maximum(std_t, 1e-5)
                std_t = np.ones(std_t.shape) * 1e-2 # *** 限制方差 ***
                for k in range(yearly_sampling_size):
                    mu_list.append([mu_t])
                    std_list.append([std_t])
        mu_list  = np.concatenate(mu_list)
        std_list = np.concatenate(std_list)
        samples  = mu_list + np.multiply(samples, std_list)
        # 
        start_idx = 0
        end_idx   = 0
        for fos in train_input:
            time_len = len(train_input[fos]) + len(train_output[fos])
            end_idx += time_len * yearly_sampling_size
            new_vecs[fos] = samples[start_idx: end_idx, :]
            start_idx = end_idx
    else:
        pass
    return new_vecs


def sampling_from_train_direction_MP(train_input, train_output, yearly_sampling_size, method, pred_model, model_name, mp_num=6):
    fos_list  = list(train_input.keys())
    total_num = len(fos_list)
    batch_num = math.ceil(total_num/ mp_num)
    start_idx = 0
    end_idx   = 0
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        end_idx    = min(start_idx + batch_num, total_num)
        fos_list_i = fos_list[start_idx: end_idx]
        train_input_i  = {fos: train_input[fos]  for fos in fos_list_i} 
        train_output_i = {fos: train_output[fos] for fos in fos_list_i} 
        start_idx  = end_idx
        results.append(pool.apply_async(sampling_from_train_direction, (train_input_i, train_output_i, yearly_sampling_size, method, )))
    pool.close()
    pool.join()
    
    new_vecs = dict()
    for res in results:
        new_vecs_i = res.get()
        for fos in new_vecs_i:
            new_vecs[fos] = new_vecs_i[fos]
    
    save_file(new_vecs, os.path.join(res_explain_path, "pred_vec_con({})({})({})".format(method, pred_model, model_name)))
    

def sampling_from_train_vector(train_input, train_output):
    # 用于检验decoder的质量
    save_file_path = os.path.join(data_file_path, model_name)
    vec_file_path  = os.path.join(save_file_path, "FoS2Vec")
    sampling_size = 10
    new_vecs = dict()
    for fos in tqdm(train_input):
        dic = read_file(os.path.join(vec_file_path, fos + ".pkl"))
        Ts = list(train_input[fos].keys()) + list(train_output[fos].keys())
        vec_arr = list()
        for t in dic:
            if t in Ts:
                vec_arr += dic[t]
        vec_arr_samples = random.sample(vec_arr, min(sampling_size, len(vec_arr)))
        new_vecs[fos] = np.array(vec_arr_samples)
    return new_vecs


def sampling_from_train_vector_MP(train_input, train_output, mp_num=8):
    fos_list  = list(train_input.keys())
    random.shuffle(fos_list)
    total_num = len(fos_list)
    batch_num = math.ceil(total_num/ mp_num)
    start_idx = 0
    end_idx   = 0
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        end_idx    = min(start_idx + batch_num, total_num)
        fos_list_i = fos_list[start_idx: end_idx]
        train_input_i  = {fos: train_input[fos]  for fos in fos_list_i} 
        train_output_i = {fos: train_output[fos] for fos in fos_list_i} 
        start_idx  = end_idx
        results.append(pool.apply_async(sampling_from_train_vector, (train_input_i, train_output_i, )))
    pool.close()
    pool.join()
    
    new_vecs = dict()
    for res in results:
        new_vecs_i = res.get()
        for fos in new_vecs_i:
            new_vecs[fos] = new_vecs_i[fos]
    return new_vecs


#%%
def clean_creative_fos(token):
    token_split  = token.split(" ")
    token_unique = set(token_split)
    token_filter = set()
    token_ = list()
    for token_k in token_split:
        if token_k not in token_filter:
            token_.append(token_k)
            token_filter.add(token_k)
    token_ = " ".join(token_)
    return token_

def count_creativefos_freq(orginalfos2creativefos, cut_t):
    # 计数creativefos的频率和由orginalfos生成          
    creativefos2freq = dict()
    for orginalfos in orginalfos2creativefos:
        creativefos_list = orginalfos2creativefos[orginalfos][: cut_t]
        for creativefos in creativefos_list:
            creativefos = clean_creative_fos(creativefos)
            if creativefos not in creativefos2freq:
                creativefos2freq[creativefos] = [0, set()]   # 频率 & 由那些fos移动产生
            creativefos2freq[creativefos][0] += 1
            creativefos2freq[creativefos][1].add(orginalfos)
    return creativefos2freq

def explain_vector_by_decoder(new_vecs_t):
    # (4.1) 读取模型的Decoder
    max_length_in = 8
    tokenizer = read_file(os.path.join(model_save_path, "tokenizer.pkl"))
    encoder   = read_file(os.path.join(model_save_path, "encoder.pkl"))
    decoder   = read_file(os.path.join(model_save_path, "decoder.pkl"))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    # (4.2) 利用Decoder解码向量
    count1, count2 = 0, 0
    SOS_token = tokenizer.encode("[SOS]")[1]
    EOS_token = tokenizer.encode("[EOS]")[1]
    orginalfos2creativefos = dict()
    for orginalfos in tqdm(new_vecs_t):
        z = new_vecs_t[orginalfos]
        z = torch.FloatTensor(z).to(device)
        # 进入encoder
        z, _ = encoder(z)
        # 进入decoder
        creativefos_list = list()
        samples_num, _ = z.shape
        start_idx = 0
        end_idx   = 0
        batch_size = 512
        batch_num  = math.ceil(samples_num / batch_size)
        for s in range(batch_num):
            end_idx = min(start_idx + batch_size, samples_num)
            decoder_hidden = z[None, start_idx:end_idx, :]
            decoder_input  = torch.tensor([[SOS_token] * (end_idx - start_idx)], device=device)  # SOS   
            decoder_input  = decoder_input.T
            
            decoded_words  = list()
            for di in range(max_length_in):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                decoded_words.append(topi)
                decoder_input = topi.squeeze().detach()
                decoder_input = decoder_input.view(-1, 1)    
            decoded_words = torch.cat(decoded_words, dim=1)
            
            for i in range(len(decoded_words)):
                fos = decoded_words[i, :, :].squeeze()
                bfos = fos != EOS_token
                creativefos_list.append(tokenizer.decode(fos[bfos]))
            start_idx = end_idx
        
        orginalfos2creativefos[orginalfos] = creativefos_list
        # 统计频率
        for fos in creativefos_list:
            if fos == orginalfos:
                count1 += 1
            else:
                count2 += 1
    print("准确率: {} / {} = {:.4f}".format(count1, count1 + count2, count1 / (count1 + count2)))
    return orginalfos2creativefos


def explain_creativefos_ratio(orginalfos2creativefos, cut_t_list):
    # cut_t_list    = np.arange(10, 200 + 10, 10)
    ratio_its_list  = list()
    ratio_itss_list = list()
    for cut_t in cut_t_list:
        creativefos2freq = count_creativefos_freq(orginalfos2creativefos, cut_t)
        
        # 评价新知识出现情况
        fos_in_full_set      = dict()
        fos_in_train_set     = dict()
        fos_in_test_set      = dict()
        fos_in_test_set_star = dict()
        for fos in creativefos2freq:
            # 在decoder训练的训练集中 (S_1 U S_2) 记作S
            if fos in fos_mu_std_nop:
                fos_in_full_set[fos] = creativefos2freq[fos]
            # 在方向预测任务的训练集中 (S_3)
            if fos in train_input:
                fos_in_train_set[fos] = creativefos2freq[fos]
            # 在(S_1 U S_2) / S_3中 (记作S_5)
            if fos in fos_mu_std_nop and fos not in train_input:
                decoded_info = creativefos2freq[fos]
                fos_freq = decoded_info[0]
                fos_time = fos_L3_BY[fos]
                fos_orgs = {fos_orginal: fos_L3_BY[fos_orginal] for fos_orginal in decoded_info[1]}
                fos_in_test_set[fos] = [(fos_freq, fos_time), fos_orgs]
            # 在(S_4)中
            if fos in fos_mu_std_nop2 and fos not in train_input:
                decoded_info = creativefos2freq[fos]
                fos_freq = decoded_info[0]
                fos_time = fos_L3_BY[fos]
                fos_orgs = {fos_orginal: fos_L3_BY[fos_orginal] for fos_orginal in decoded_info[1]}
                fos_in_test_set_star[fos] = [(fos_freq, fos_time), fos_orgs]
        
        ratio_its  = len(fos_in_test_set)      / (len(fos_mu_std_nop) - len(train_input))
        ratio_itss = len(fos_in_test_set_star) / (len(fos_mu_std_nop2) - len(train_input))
        ratio_its_list.append(ratio_its)
        ratio_itss_list.append(ratio_itss)
        
        print("------")
        print("预测长度: {}; 主题数目: {}".format(cut_t, len(creativefos2freq)))
        print("(S_F x S)  / S   = {} / {} = {:.6f}".format(len(fos_in_full_set),      len(fos_mu_std_nop),                     len(fos_in_full_set)     /len(fos_mu_std_nop)))
        print("(S_F x S_3)/ S_3 = {} / {} = {:.6f}".format(len(fos_in_train_set),     len(train_input),                        len(fos_in_train_set)    /len(train_input)))
        print("(S_F x S_5)/ S_5 = {} / {} = {:.6f}".format(len(fos_in_test_set),      len(fos_mu_std_nop)  - len(train_input), len(fos_in_test_set)     /(len(fos_mu_std_nop)  - len(train_input))))
        print("(S_F x S_4)/ S_4 = {} / {} = {:.6f}".format(len(fos_in_test_set_star), len(fos_mu_std_nop2) - len(train_input), len(fos_in_test_set_star)/(len(fos_mu_std_nop2) - len(train_input))))
        print("------")
    
    fos_not_in_test_set_star = dict()
    for fos in fos_mu_std_nop2:
        if fos not in train_input and fos not in fos_in_test_set:
            fos_not_in_test_set_star[fos] = fos_mu_std_nop2[fos]
    return (ratio_its_list, ratio_itss_list), (fos_in_test_set, fos_in_test_set_star, fos_not_in_test_set_star)


def plot_explain_ratio(orginalfos2creativefos_p, orginalfos2creativefos_t, cut_t, model_name, yticks):
    cut_t_list = np.arange(10, cut_t + 10, 10)
    (ratio_its_list, ratio_itss_list), (fos_in_test_set, fos_in_test_set_star, fos_not_in_test_set_star) = explain_creativefos_ratio(orginalfos2creativefos_p, cut_t_list)
    (ratio_its_t, ratio_itss_t), (_, _, _) = explain_creativefos_ratio(orginalfos2creativefos_t, cut_t_list)

    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    # createtivefos_p 在预测时段内比率
    plt.plot(cut_t_list, ratio_its_list, c='blue', marker='+', markersize=10, linewidth=1, label=r"$|S_F \cap S_5|$/$|S_5|$")
    plt.plot(cut_t_list, ratio_itss_list, c='red', marker='*', markersize=10, linewidth=1, label=r"$|S_F \cap S_4|$/$|S_4|$")
    # createtivefos_t 在预测时段内比率
    plt.plot(cut_t_list, np.ones(len(cut_t_list)) * ratio_its_t, c='gray', marker='+', markersize=10, linewidth=1, label=r"$|S_B \cap S_5|$/$|S_5|$")
    plt.plot(cut_t_list, np.ones(len(cut_t_list)) * ratio_itss_t, c='gray', marker='*', markersize=10, linewidth=1, label=r"$|S_B \cap S_4|$/$|S_4|$")
    plt.xlabel("T")
    plt.ylabel("Ratio")
    plt.legend(frameon=False, loc='upper left')
    plt.xticks(np.arange(0, cut_t+cut_t/10, cut_t/10))
    if len(yticks):
        plt.yticks(yticks)
    if model_name == 'bert': model_name = 'bert-base-uncased'
    plt.title(model_name)


def plot_explain_evolution_process(orginalfos2creativefos_p, fos):
    creativetoken_list = orginalfos2creativefos_p[fos] 
    # 确定creativefos, 剔除其中叠词
    token2t = dict()
    for i, token in enumerate(creativetoken_list):
        token_ = clean_creative_fos(token)
        if token_ not in token2t:
            print(token, "->", token_)
            token2t[token_] = i
    # 修正错误词
    fix_dict = {"bottleneckental": "bottleneck",
                "improvementental": "improvement",
                "dataental": "data"}
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)

    for j, token in enumerate(token2t):
        
        if token in train_input or token in fos_mu_std_nop:  # F看见过成主题 & F未见过的主题
            pass
        else:
            continue
        
        t = token2t[token]
        y = min(fos_mu_std_nop[token].keys())
     
        if token in train_input: 
           c = 'black'
           plt.scatter([t], [y], c=c, marker='o', s=15)
        elif token in fos_mu_std_nop:
           c = 'red'
           plt.scatter([t], [y], c=c, marker='*', s=15)
        
        # 主题名字
        token_ = list()
        for token_k in token.split(" "):
            if token_k in fix_dict:
                token_k = fix_dict[token_k]
            token_.append(token_k)
        token = " ".join(token_)
        if t <50:
            ha = 'left'
        else:
            ha = 'center'
        plt.text(t, y, token, ha=ha, wrap=True, fontsize=10, c=c)
        
    plt.yticks(np.arange(1900, 2040, 20))
    plt.xlabel("T")
    plt.ylabel(r"$t_{born}$")
    plt.xticks(np.arange(0, 350, 50))
    plt.xlim(-10, 320)
    plt.title(fos)
    
#%%
if __name__ == "__main__":
    #  Section 3.4 Explaining the motion of topic emebeddings
    def main():
        # (0) 读取数据
        model_name = "all-MiniLM-L6-v2"
        # model_name = "all-mpnet-base-v2"
        # model_name = 'bert'
        model_save_path = os.path.join(data_file_path, model_name)
        filtered_nop2   = 500
        train_input, train_output, test_output, fos2idx, fos_mu_std_nop, fos_mu_std_nop2 = generate_prediction_data(data_file_path, model_name, filtered_nop2)
        fos_L3_BY = read_file(os.path.join(data_file_path, "fos_L3_BY.pkl")) 
        
        # (2) 预测向量的均值和方差
        cut_t = 300
        var_predict(train_input, train_output, cut_t, model_name)
    
        # (3) 利用预测的均值采样生成向量
        yearly_sampling_size = 1
        method = 'sampling'  # mean # sampling
        pred_model = 'var'
        
        # original fos -> creative fos in the experimental group: 不可生成fos: 说明预测质量有问题
        sampling_from_pred_direction_MP(yearly_sampling_size, method, pred_model, model_name)
        tmp_path = os.path.join(res_explain_path, "pred_vec_exp({})({})({})".format(method, pred_model, model_name))
        ofos2cfos_exp = explain_vector_by_decoder(read_file(tmp_path))
        save_file(ofos2cfos_exp, "./Explain/orginalfos2creativefos_p({})({})({}).pkl".format(method, pred_model, model_name))
        
        # original fos -> creative fos in the control group: 预测训练集中fos的类内均值可生成目标fos: 说明fos的类内均值可表示该fos
        sampling_from_train_direction_MP(train_input, train_output, yearly_sampling_size, method, pred_model, model_name)
        tmp_path = os.path.join(res_explain_path, "pred_vec_con({})({})({})".format(method, pred_model, model_name))
        ofos2cfos_con = explain_vector_by_decoder(read_file(tmp_path))
        save_file(ofos2cfos_con, "./Explain/orginalfos2creativefos_t({})({})({}).pkl".format(method, pred_model, model_name))
          
        # 用于检测decoder的质量: 是否能将fos的vec解码回fos
        # new_vecs_d = sampling_from_train_vector_MP(train_input, train_output)
        # save_file(new_vecs_d, "./temp/new_vecs_d") 
        
        # (4)
        # 读取预测向量生成的主题词 & 读取训练向量生成的主题词
        orginalfos2creativefos_t = read_file("./Explain/orginalfos2creativefos_t({})({})({}).pkl".format(method, pred_model, model_name))
        orginalfos2creativefos_p = read_file("./Explain/orginalfos2creativefos_p({})({})({}).pkl".format(method, pred_model, model_name))
        
        # 绘制解析比率
        yticks = np.arange(0, 0.12, 0.02)
        plot_explain_ratio(orginalfos2creativefos_p, orginalfos2creativefos_t, cut_t, model_name, [])
       
        # 挑选解释案例
        cut_t_list = np.arange(10, cut_t + 10, 10)
        (ratio_its_list, ratio_itss_list), (fos_in_test_set, fos_in_test_set_star, fos_not_in_test_set_star) = explain_creativefos_ratio(orginalfos2creativefos_p, cut_t_list)
    
        fos = 'language analysis' # 'vehicle dynamics' # 'travelling salesman problem' # 'database query' # 'language analysis'
        plot_explain_evolution_process(orginalfos2creativefos_p, fos)
