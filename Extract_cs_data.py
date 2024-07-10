#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:20:58 2023

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
import multiprocessing
import pandas as pd
import autograd.numpy as np
import prettytable as pt
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

import gensim
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric

from utils import *


#%%
def extract_fos_parent_children_info():
    """ 从FieldsOfStudy.nt 和 FieldOfStudyChildren.nt中抽取FoS的上下位信息 """
    tmp1 = list()
    tmp2 = list()
    tmp3 = list()
    count = 0
    with open(Fos_file_1, 'r') as f:
        while True:
            oneline = f.readline().strip()
            if oneline:
                oneline_split = oneline.split(" ")
                if oneline_split[1] == "<http://xmlns.com/foaf/0.1/name>":    
                    fid  = re.findall("\d+", oneline)[0]
                    name = re.findall(r'"([^"]+)"', oneline)[0]
                    tmp1.append(fid)
                    tmp2.append(name)
                    # print(oneline, name, fid)
                if oneline_split[1] == "<http://ma-graph.org/property/level>":            
                    level = re.findall(r'"([^"]+)"', oneline)[0]
                    tmp3.append(level)
                    # print(oneline, level)
            else:
                break
    # field of study 的 level 和 name
    FoSs_info = dict()
    for fid, name, level in zip(tmp1, tmp2, tmp3):
        FoSs_info[fid] = (name, int(level))
    
    # field of study 的 has parent (上下位关系)
    FoSs_childs = list()
    with open(Fos_file_2, 'r') as f:
        while True:
            oneline = f.readline().strip()
            if oneline:
                oneline_split = oneline.split(" ")
                child  = oneline_split[0]
                parent = oneline_split[2]
                
                child_id  = re.findall("\d+", child)[0]
                parent_id = re.findall("\d+", parent)[0]
                FoSs_childs.append((child_id, parent_id))
            else:
                break
    FoSs_childs_ = dict()
    for child, parent in FoSs_childs:
        if parent not in FoSs_childs_:
            FoSs_childs_[parent] = list()
        FoSs_childs_[parent].append(child)
    FoSs_childs  = FoSs_childs_
    
    with open("./temp/FoSs_info.pkl", 'wb') as f:
        pickle.dump(FoSs_info, f)
    with open("./temp/FoSs_childs.pkl", 'wb') as f:
        pickle.dump(FoSs_childs, f)
        
        
def select_fos_level_from_field(level0='Computer science'):
    """ 从特定领域挑选特定LEVEL的FoS主题构成研究对象 """
    
    def find_childs(fid, FoSs_childs):
        # 递归寻找fid下所有的下级fos
        all_childs_fid  = list()
        if fid not in FoSs_childs:  # 该fid无下级fid
            return all_childs_fid
        else:
            childs_fid      = FoSs_childs[fid]
            all_childs_fid += childs_fid
            for fid_i in childs_fid:
                childs_fid_i    = find_childs(fid_i, FoSs_childs)
                all_childs_fid += childs_fid_i
            return all_childs_fid
    
    if os.path.exists("./temp/FoSs_info.pkl") and os.path.exists("./temp/FoSs_childs.pkl"):
        FoSs_info   = read_file("./temp/FoSs_info.pkl")
        FoSs_childs = read_file("./temp/FoSs_childs.pkl")
    else:
        extract_fos_parent_children_info()
        FoSs_info   = read_file("./temp/FoSs_info.pkl")
        FoSs_childs = read_file("./temp/FoSs_childs.pkl")

    level2name = dict()
    name2fid   = dict()
    for fid in FoSs_info:
        name, level = FoSs_info[fid]
        name2fid[name] = fid
        if level not in level2name:
            level2name[level] = list()
        level2name[level].append(name)
    
    # 挑选特定领域level0下FoS
    level0_id = name2fid[level0]
    all_childs_fid = find_childs(level0_id, FoSs_childs)
    all_childs_fid = {fid: "" for fid in all_childs_fid}
    
    level2name_cs = dict()
    for fid in FoSs_info:
        if fid not in all_childs_fid:
            continue
        name, level = FoSs_info[fid]
        if level not in level2name_cs:
            level2name_cs[level] = list()
        level2name_cs[level].append(name)
    
    tb = pt.PrettyTable()
    tb.title = level0
    tb.field_names = ["MAG-LEVEL", "总数目", "(领域)数目"]
    for level in range(0, 6):
        if level == 0:
            tb.add_row([level, len(level2name[level]), 1])
        else:
            tb.add_row([level, len(level2name[level]), len(level2name_cs[level])])
    print(tb)
     
    # 挑选特定领域level0下FoS_L1和其子主题FoS_L2
    has_childs_count = 0
    l = 1
    fos_L2_dict = dict()
    for fos_L2 in level2name_cs[l]:
        fid_L2 = name2fid[fos_L2]
        # fos_L2下的fos_L3
        if fid_L2 in FoSs_childs:
            fos_L2_childs = list()
            for fid_L2_child in FoSs_childs[fid_L2]:
                fos_L2_child, level = FoSs_info[fid_L2_child]
                fos_L2_child = fos_L2_child.lower()
                if level == l + 1:
                    fos_L2_childs.append(fos_L2_child)
            if len(fos_L2_childs) != 0:
                fos_L2 = fos_L2.lower()
                fos_L2_dict[fos_L2] = fos_L2_childs
                has_childs_count += 1            

    return  fos_L2_dict


#%%
def get_fos_L3_bornyear(fos_L3_NoP, acc_nop=10):
    # 从fos_L3_NoP中抽取所有fos_L3至少发表10篇论文的年份作为主题诞生的年份
    fos_L3_BY = dict()
    for fos_L3 in fos_L3_NoP:
        if len(fos_L3_NoP[fos_L3]) > 0:
            # 累计发文量超过阈值
            Ts = sorted(fos_L3_NoP[fos_L3].keys())
            accNoP = 0
            for t in Ts:
                accNoP += fos_L3_NoP[fos_L3][t]
                # 累计发文量超过10
                if accNoP >= acc_nop:
                    break
            fos_L3_BY[fos_L3] = t
    return fos_L3_BY  


def get_fos_L3_bornyear_from_fos_L2(fos_L3_BY, fos_L2_dict, fos_L2):
    # 获取fos_L2下的所有fos_L3
    if fos_L2 == "all":
        # 遍历所有fos_L2
        T2fos_L3 = dict()
        for fos_L2_i in fos_L2_dict:
            for fos_L3 in fos_L2_dict[fos_L2_i]:
                fos_L3 = fos_L3.lower()
                if fos_L3 in fos_L3_BY:
                    t = fos_L3_BY[fos_L3]
                    if t not in T2fos_L3:
                        T2fos_L3[t] = [fos_L3]
                    else:
                        T2fos_L3[t].append(fos_L3)
    else:
        # 遍历特定fos_L2
        T2fos_L3 = dict()
        for fos_L3 in fos_L2_dict[fos_L2]:
            fos_L3 = fos_L3.lower()
            if fos_L3 in fos_L3_BY:
                t = fos_L3_BY[fos_L3]
                if t not in T2fos_L3:
                    T2fos_L3[t] = [fos_L3]
                else:
                    T2fos_L3[t].append(fos_L3)
    return T2fos_L3


def plot_number_of_new_topics(fos_L2_dict, fos_L3_BY, fos_L2=''):
    
    if fos_L2 not in fos_L2_dict and fos_L2 != "all":
        # 随机挑选FoS_L2
        fos_L2 = random.sample(fos_L2_dict.keys(), 1)[0]
    
    # 逐年的新主题数目
    T2fos_L3 = get_fos_L3_bornyear_from_fos_L2(fos_L3_BY, fos_L2_dict, fos_L2)

    T_dis = dict()
    for t in T2fos_L3:
        T_dis[t] = len(T2fos_L3[t])
         
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)

    X = sorted(T_dis.keys())
    Y = [T_dis[x] for x in X]
    plt.scatter(X, Y, marker='*', c='black')
    plt.xlim(1900, 2025)
    plt.xlabel("Time")
    plt.ylabel("Number of new topics")
    plt.title(fos_L2)


def plot_accumulative_freq_distribution(fos_L3_NoP):
    nop2not = dict()
    for fos in fos_L3_NoP:
        nop = sum([fos_L3_NoP[fos][t] for t in fos_L3_NoP[fos]])
        if nop not in nop2not:
            nop2not[nop] = 1
        else:
            nop2not[nop] += 1
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    X = sorted(list(nop2not.keys())) 
    Y = [nop2not[x] for x in X]
    plt.scatter(X, Y, linewidth=.5, c='black', s=10, marker="+")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Number of papers studying a topic")
    plt.ylabel(r"Number of topics $(FoS_{L2})$")


#%%
def create_train_data(fos_L2, T0, T1, T2, save_data_path):
    """ 生成论文向量表征所需数据 """
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
    
    fos_L2_dict = select_fos_level_from_field('Computer science')
    if not os.path.exists(os.path.join(save_data_path, "fos_L3_NoP.pkl")):
        # 统计FoS_L3逐年的频率
        fos_L3_NoP  = dict()
        for fos_L2_ in fos_L2_dict:
            for fos_L3 in fos_L2_dict[fos_L2_]:
                fos_L3 = fos_L3.lower()
                fos_L3_NoP[fos_L3] = dict()
        # 
        for i in tqdm(range(0, 17)):
            with open(file_fos_i.format(i), 'r') as f:
                while True:
                    oneline = f.readline().strip()
                    if oneline:
                        oneline_json = json.loads(oneline)
                        year = oneline_json['t']
                        pid  = oneline_json['pid']
                        FoSs = oneline_json['f']
                        if year == '':
                            continue
                        for fos_i in FoSs:
                            fos_i = fos_i.lower()
                            if fos_i in fos_L3_NoP:
                               if year not in fos_L3_NoP[fos_i]:
                                   fos_L3_NoP[fos_i][year] = 1
                               else:
                                   fos_L3_NoP[fos_i][year] += 1
                    else:
                        break
        # 无统计频率剔除
        del_fos = set()
        for fos in fos_L3_NoP:
            if len(fos_L3_NoP[fos]) == 0:
                del_fos.add(fos)
        for fos in del_fos:
            del fos_L3_NoP[fos]
                    
        save_file(fos_L3_NoP, os.path.join(save_data_path, "fos_L3_NoP.pkl"))
    else:
        # 读取FoS_L3逐年的频率
        fos_L3_NoP = read_file(os.path.join(save_data_path, "fos_L3_NoP.pkl"))  
    
    # 选择一个fos_L2, 进行后续分析
    data_file_path = os.path.join(save_data_path, fos_L2)
    if not os.path.exists(data_file_path):
        print("创建文件夹:", data_file_path)
        os.mkdir(data_file_path)
    else:
        print("存在文件夹:", data_file_path)
    
    # FoS_L3的出生年份
    fos_L3_BY = get_fos_L3_bornyear(fos_L3_NoP)
    T2fos_L3  = get_fos_L3_bornyear_from_fos_L2(fos_L3_BY, fos_L2_dict, fos_L2)  # 只要特定fos_L2下fos_L3
    plot_number_of_new_topics(fos_L2_dict, fos_L3_BY, fos_L2)
    focal_fos_L3 = set() 
    for t in T2fos_L3:
        for fos_L3 in T2fos_L3[t]:
            focal_fos_L3.add(fos_L3)
    other_fos_L3 = set()
    for fos in fos_L3_BY:
        if fos not in focal_fos_L3:
            other_fos_L3.add(fos)
    save_file(fos_L3_BY,    os.path.join(data_file_path, "fos_L3_BY.pkl"))     # 每个fos的出生年 (至少10篇论文)
    save_file(focal_fos_L3, os.path.join(data_file_path, "focal_fos_L3.pkl"))  # 每个fos的出生年 (至少10篇论文)
    save_file(other_fos_L3, os.path.join(data_file_path, "other_fos_L3.pkl"))  # 每个fos的出生年 (至少10篇论文)
    
    tb = pt.PrettyTable()
    tb.field_names = ["主题", "数目"]
    tb.add_row(["所有数目", len(fos_L3_BY)])
    tb.add_row(["focal主题数", len(focal_fos_L3)])
    tb.add_row(["other主题数", len(other_fos_L3)])
    print(tb)
    
    # 准备训练数据
    for i in range(0, 17):
        # 读取FoSs   
        pid2FoSs = dict()
        with open(file_fos_i.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    year = oneline_json['t']
                    pid  = oneline_json['pid']
                    FoSs = oneline_json['f']
                    if year == '':
                        continue
                    # 过滤后只保留内部FoS_L3
                    FoSs_ = list()
                    for fos in FoSs:
                        if fos in focal_fos_L3:  
                            if year >= fos_L3_BY[fos]:        # 自该fos已诞生, 才统计该论文 (至少已发表10篇)
                                FoSs_.append(fos)
                    if len(FoSs_) > 0:
                        pid2FoSs[pid] = (FoSs_, year)         
                else:
                    break
        # 读取标题 + 摘要
        non_english_titles = list()
        pid2content = dict()
        with open(file_content_i.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    pid = oneline_json['pid']                 # 论文id
                    if pid not in pid2FoSs:
                        continue
                    content  = oneline_json["content"].strip()
                    con_spl  = content.split(";")
                    year     = con_spl[0].strip()             # 出版年份
                    title    = con_spl[1].strip()             # 标题
                    abstract = ";".join(con_spl[2:]).strip()  # 摘要
                    if len(title) == 0 and len(abstract) == 0:
                        continue
                    content  = title # + "." + abstract
                    
                    remove_chars = '[×¿®☆∆·․´∗￫™′°’/／„«»‑—－−―–‐½∞!√²"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
                    content  = re.sub(remove_chars, " ", content)
                
                    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]
                    content = preprocess_string(content, CUSTOM_FILTERS)
                    
                    content1 = " ".join(content).strip()
                    content2 = "".join(content).strip()
                    if len(content) == 0:                    # 长度为0
                        continue
                    if not content2.isalpha():               # 非英语标题
                        non_english_titles.append(content2)
                        continue
                    pid2content[pid] = content1
                else:
                    break
        # 合并信息
        pid2str = dict()
        for pid in pid2FoSs:
            if pid in pid2content:
                content    = pid2content[pid]
                FoSs, year = pid2FoSs[pid]
                # 选取T0-T2之间的论文
                if T0 <= year and year <= T2:  
                    pid2str[pid] = (FoSs, content, year)
        print("({}) {}-{}年间的论文数目: {}/{}/{}".format(i, T0, T2, len(pid2str), len(pid2FoSs), len(pid2content))) 
        del pid2FoSs, pid2content
        # 储存
        pid2str_path_i = os.path.join(data_file_path, "pid2str({}).pkl".format(i))
        save_file(pid2str, pid2str_path_i)
        

def create_fos_co_matrix():
    """ 抽取fos之间的逐年共现在关系 """
    comatrix = dict()
    for i in range(0, 17):
        pid2str_path_i = os.path.join(data_file_path, "pid2str({}).pkl".format(i))
        pid2str = read_file(pid2str_path_i)
        for pid in tqdm(pid2str):
            FoSs, content, year = pid2str[pid]
            for fos_i in FoSs:
                if fos_i not in comatrix:
                    comatrix[fos_i] = dict()
                if year not in comatrix[fos_i]:
                    comatrix[fos_i][year] = dict()
                for fos_j in FoSs:
                    if fos_i == fos_j:
                        continue
                    else:
                        if fos_j not in comatrix[fos_i][year]:
                            comatrix[fos_i][year][fos_j] = 1
                        else:
                            comatrix[fos_i][year][fos_j] += 1
    save_file(comatrix, os.path.join(data_file_path, "comatrix"))
    



#%%
if __name__ == "__main__":
    # 从OAG官网下载的原始数据: http://www.aminer.cn/oag-2-1
    # 根据fos == computer science筛选处cs领域论文, 并储存在mag_papers_{}.txt
    Fos_file_1     = "/mnt/disk2/MAG_DATA_SET/FieldsOfStudy.nt"
    Fos_file_2     = "/mnt/disk2/MAG_DATA_SET/FieldOfStudyChildren.nt"
    file_content_i = "/mnt/disk2/MAG_DATA_SET/MAGv2.1-abstract-cs/mag_papers_{}.txt"
    file_fos_i     = "/mnt/disk2/MAG_DATA_SET/MAGv2.1-meta-computer science/mag_papers_{}.txt"
    
    # 根据Fos_file_2, 挑选特定 FoS_Level
    fos_L2 = 'all'
    T0 = 1900
    T1 = 2000
    T2 = 2018
    save_data_path = "/mnt/disk3/NewTopicsPrediction"    
    data_file_path = os.path.join(save_data_path, fos_L2)
    # 
    def main():
        create_train_data(fos_L2, T0, T1, T2, save_data_path)
        create_fos_co_matrix()
