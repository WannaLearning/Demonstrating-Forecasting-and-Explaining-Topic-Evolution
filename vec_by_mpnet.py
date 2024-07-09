#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:36:07 2023

@author: aixuexi
"""
import os 
import pickle
import re
import json
import random
import math
import time
import multiprocessing
import pandas as pd
import autograd.numpy as np
import prettytable as pt
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from tqdm import tqdm
from scipy import spatial

import gensim
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer

from extract_cs_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def mpnet_encoder():
    model_name = "all-mpnet-base-v2"
    # model_name = "all-MiniLM-L6-v2"
    pretrained_path = os.path.join("/mnt/disk2", model_name)
        
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = AutoModel.from_pretrained(pretrained_path)
    model.to(device)
    model.eval()
    
    save_file_path = os.path.join(data_file_path, model_name)
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)

    # 参数配置
    batch_size  = 256
    max_length  = 30  # bert model truncation length
    for i in range(13, 14):
        # 读数据
        pid2str = read_file(os.path.join(data_file_path, "pid2str({}).pkl".format(i)))
        
        # BERT 的输入
        batch_sen  = list()   # fos + "[SEP]" + content # content = title + "." + abstract
        batch_fos  = list()   # fos 
        batch_year = list()   # publication year
        # 生成的向量
        results = dict()  # fos -> year -> vec 
        c = -1
        for pid in tqdm(pid2str):
            c += 1
            FoSs, content, year = pid2str[pid]
            for fos in FoSs:
                batch_sen.append([fos, content])
                batch_fos.append(fos)
                batch_year.append(year)
            #
            if len(batch_sen) >= batch_size or c == len(pid2str)-1:
                # Tokenize sentences
                encoded_input = tokenizer(batch_sen, padding=True, truncation=True, return_tensors='pt')
                encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
                encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)
                if 'token_type_ids' in encoded_input:
                    encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(device)

                with torch.no_grad():
                    model_output = model(**encoded_input) 
                
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                # Normalize embeddings
                batch_vec = F.normalize(sentence_embeddings, p=2, dim=1)
                batch_vec = batch_vec.cpu().numpy()
                
                # 存放batch内向量结果
                for j in range(len(batch_fos)):
                    fos  = batch_fos[j]
                    year = batch_year[j]
                    vec  = batch_vec[j]
                    if fos not in results:
                        results[fos] = dict()
                    if year not in results[fos]:
                        results[fos][year] = list()
                    results[fos][year].append(np.array(vec, dtype=(np.float16)))    
                    
                # 清空缓存
                batch_sen    = list()
                batch_fos    = list()
                batch_year   = list()
                torch.cuda.empty_cache()
        # 储存
        del pid2str
        save_file(results, os.path.join(save_file_path, "vec_{}.pkl".format(i)))
        
    # 分fos存储
    store_vector_by_fos(save_file_path)   


def prepare_filter_fos_by_freq(freq_threshold = 100):
    # 只挑选frequency超过阈值的fos来训练decoder
    # 过滤后的fos才是我们关注的重点, 进而避免:
    # (1) 低频fos对decoder精度的影响; (2) 低频fos在decoder进行解释时的随机性干扰, 因为低频词是随机产生
    model_name = "all-MiniLM-L6-v2"
    save_file_path = os.path.join(data_file_path, model_name)
    file_path = os.path.join(save_file_path, "FoS2Vec")

    filtered_fos   = dict()
    for fos in tqdm(os.listdir(file_path)):
        dic = read_file(os.path.join(file_path, fos))
        nop = 0
        for year in dic:
            nop_j = len(dic[year])
            nop  += nop_j
        if nop >= freq_threshold:
            fos = fos[:-4]
            filtered_fos[fos] = nop
    
    # 这些fos被训练decoder, 即等待被预测与解释的主题
    save_file(filtered_fos, os.path.join(data_file_path, "filtered_fos_for_decoder.pkl"))
    return filtered_fos
    

#%%
def KL_with_standard_gaussian_prior(mu, log_sigma_squared):
    kl_distance = -0.5 * torch.sum(1 + log_sigma_squared - mu.pow(2) - log_sigma_squared.exp(), dim=1)
    kl_regular  = kl_distance / mu.shape[-1]
    return kl_regular


class EncoderForMPNET(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderForMPNET, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.proj1 = nn.Linear(self.input_dim,  self.hidden_dim)
        self.proj2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.proj3 = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        x  = torch.relu(self.proj1(x))
        mu = self.proj2(x)
        log_sigma_squared = self.proj3(x)
        return mu, log_sigma_squared


class DecoderForMPNET(nn.Module):
    
    def __init__(self, hidden_size_gru, hidden_size_emb, output_size, bert_embeddings_weight, trainable):
        # output_size: 字典大小
        super(DecoderForMPNET, self).__init__()
        self.hidden_size_emb = hidden_size_emb
        self.hidden_size_gru = hidden_size_gru
        self.output_size = output_size
        self.embedding = nn.Embedding(num_embeddings = output_size, 
                                      embedding_dim  = hidden_size_emb, 
                                      _weight = bert_embeddings_weight)
        if trainable: 
            self.embedding.requires_grad = True
        else:
            self.embedding.requires_grad = False  
        self.gru = nn.GRU(hidden_size_emb, hidden_size_gru, batch_first=True)
        self.proj1 = nn.Linear(hidden_size_gru, hidden_size_gru)
        self.proj2 = nn.Linear(hidden_size_gru, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.gru(output, hidden)
        output = self.proj2(torch.relu(self.proj1(output)))
        output = torch.nn.functional.log_softmax(output, dim=-1)
        return output, hidden


def train_EncDec_For_Explain():
    global data_file_path
    
    # model_name = "all-MiniLM-L6-v2"
    model_name = "all-mpnet-base-v2"
    pretrained_path = os.path.join("/mnt/disk2",   model_name)
    model_save_path = os.path.join(data_file_path, model_name)
    save_file_path  = model_save_path
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = AutoModel.from_pretrained(pretrained_path)
    # 添加两个特殊字符
    SOS_token = "[EOS]"
    EOS_token = "[SOS]"
    tokenizer.add_tokens([SOS_token, EOS_token])
    model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.encode(SOS_token)[1]
    
    # 冻结参数
    for name, params in model.named_parameters():  
        if name == "embeddings.word_embeddings.weight":
            params.requires_grad = False
            bert_embeddings_weight = params.data
        else:
            params.requires_grad = False
        break
    
    bert_dim = 256
    encoder = EncoderForMPNET(768, bert_dim, bert_dim)  # 384 for all-MiniLM-L6-v2; 784 for all-mpnet-base-v2
    decoder = DecoderForMPNET(bert_dim, bert_embeddings_weight.shape[1], bert_embeddings_weight.shape[0], bert_embeddings_weight, True)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    CEloss        = torch.nn.CrossEntropyLoss(reduction='none')
    learning_rate = 1e-3
    optimizer     = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)                             
    batch_size    = 256
    max_length_in = 10
    epochs        = 1
    
    # 读取我们关注的超过频率阈值的fos, 其由prepare_filter_fos_by_freq函数获取
    tmp_path = os.path.join(data_file_path, "filtered_fos_for_decoder.pkl")
    if os.path.exists(tmp_path):
        filtered_fos = read_file(tmp_path)
    else:
        filtered_fos = prepare_filter_fos_by_freq()
    
    for e in range(epochs):
        right_counts = 0
        batch_counts = 0
        for i in range(17):
            # 读取向量
            results  = read_file(os.path.join(model_save_path, "vec_{}.pkl".format(i)))
            vec_pool = list()
            for fos in results:
                # 低频次的fos_L3被过滤掉
                if fos not in filtered_fos:
                    continue
                for year in results[fos]:
                    for vec in results[fos][year]:
                        vec_pool.append((fos, vec))
            del results
            # 洗牌向量
            random.shuffle(vec_pool)
            start_idx = 0
            end_idx   = 0
            batch_num = math.ceil(len(vec_pool) / batch_size)
            for j in range(batch_num):
                optimizer.zero_grad()
                # 将读取的向量划分为batch输入encoder和decoder
                end_idx    = min(start_idx + batch_size, len(vec_pool))
                batch_data = vec_pool[start_idx: end_idx]
                batch_enc_in  = list() # vector
                batch_dec_in  = list() # [SOS] + fos
                batch_dec_out = list() # fos + [EOS]
                for fos, vec in batch_data:
                    batch_enc_in.append(vec)
                    batch_dec_in.append("[SOS] " + fos)
                    batch_dec_out.append(fos + " [EOS]")
                batch_enc_in       = torch.tensor(batch_enc_in, dtype=torch.float32).to(device)
                input_for_in       = tokenizer(batch_dec_in,  
                                               padding='max_length', 
                                               max_length=max_length_in,
                                               truncation=True, 
                                               return_tensors='pt', 
                                               add_special_tokens=False)
                input_ids_in       = torch.tensor(input_for_in['input_ids']).to(device)
                attention_mask_in  = torch.tensor(input_for_in['attention_mask']).to(device)
                # token_type_ids_in  = torch.tensor(input_for_in['token_type_ids']).to(device)
                input_for_out      = tokenizer(batch_dec_out, 
                                               padding='max_length', 
                                               max_length=max_length_in, 
                                               truncation=True, 
                                               return_tensors='pt', 
                                               add_special_tokens=False)
                input_ids_out      = torch.tensor(input_for_out['input_ids']).to(device)
                attention_mask_out = torch.tensor(input_for_out['attention_mask']).to(device)
                # token_type_ids_out = torch.tensor(input_for_out['token_type_ids']).to(device)
                
                # ENCODER
                mu, log_sigma_squared = encoder(batch_enc_in)
                
                # discontinus (简单的autoencoder)
                kl_div = 0
                z = mu
                
                # continus (vae)
                # kl_div = KL_with_standard_gaussian_prior(mu, log_sigma_squared) # KL distance
                # e = torch.empty_like(mu).normal_(mean=0., std=1.)
                # z = mu + e * torch.sqrt(torch.exp(log_sigma_squared))
                
                # DECODER
                reconstruction_loss   = 0
                teacher_forcing_ratio = 0.5
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    # Teacher forcing: Feed the target as the next input
                    output = list()
                    decoder_hidden = z[None, :, :]
                    for di in range(max_length_in):
                        decoder_input = input_ids_in[:, di: di+1]
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                        output.append(decoder_output)
                    output = torch.cat(output, axis=1)       
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    output = list()
                    decoder_input  = input_ids_in[:, 0: 1]
                    decoder_hidden = z[None, :, :]
                    for di in range(max_length_in):
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze(axis=-1).detach()  # detach from history as input
                        output.append(decoder_output)
                    output = torch.cat(output, axis=1)
                    
                # 判断预测精度
                right_count = 0
                batch_count = 0
                topv, topi = output.data.topk(1)
                for k in range(len(output)):
                    batch_dec_out_pred_k = tokenizer.decode(topi.squeeze()[k])
                    batch_dec_out_real_k = batch_dec_out[k]
                    seq_pred = batch_dec_out_pred_k.split(" ")
                    seq_real = batch_dec_out_real_k.split(" ")[:-1]
                    count = 1
                    for l in range(len(seq_real)):
                        if seq_pred[l] != seq_real[l]:
                            count = 0
                            break
                    right_count += count
                    batch_count += 1
                right_counts += right_count
                batch_counts += batch_count
                prec_ratio = right_counts / batch_counts
                      
                # 计算loss
                output_t    = torch.transpose(output, 1, 2)
                unmask_loss = CEloss(output_t, input_ids_out)
                mask_loss   = torch.multiply(unmask_loss, attention_mask_out)
                mask_loss   = torch.divide(torch.sum(mask_loss, axis=-1), torch.sum(attention_mask_out, axis=-1))
                reconstruction_loss += mask_loss
                # 梯度下降
                elbo = - reconstruction_loss - kl_div
                loss = - elbo.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
                optimizer.step()
                # 输出loss
                print("Epoch {} File {} Bacth {} / {}: {:.6f} ({:.6f})".format(e+1, i, j, batch_num, loss, prec_ratio))
                # 清空缓存
                batch_enc_in  = list() # vector
                batch_dec_in  = list() # [SOS] + fos
                batch_dec_out = list() # fos + [EOS]
                torch.cuda.empty_cache()
                start_idx  = end_idx
                # break
            print("储存模型")
            save_file(tokenizer, os.path.join(model_save_path, "tokenizer.pkl"))
            save_file(encoder,   os.path.join(model_save_path, "encoder.pkl"))
            save_file(decoder,   os.path.join(model_save_path, "decoder.pkl"))
        

def evaluate_EncDec():
    global data_file_path
    
    model_name = "all-MiniLM-L6-v2"
    # model_name = "all-mpnet-base-v2"
    model_save_path = os.path.join(data_file_path, model_name)
    max_length_in = 10
    tokenizer = read_file(os.path.join(model_save_path, "tokenizer.pkl"))
    encoder   = read_file(os.path.join(model_save_path, "encoder.pkl"))
    decoder   = read_file(os.path.join(model_save_path, "decoder.pkl"))

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    encoder.eval()
    decoder.eval()
    
    # 读取我们关注的超过频率阈值的fos, 其由prepare_filter_fos_by_freq函数获取
    tmp_path = os.path.join(data_file_path, "filtered_fos_for_decoder.pkl")
    if os.path.exists(tmp_path):
        filtered_fos = read_file(tmp_path)
    else:
        filtered_fos = prepare_filter_fos_by_freq()
    
    model_save_path = os.path.join(data_file_path, model_name)
    batch_size   = 256
    max_length_in = 10
    right_counts = 0
    batch_counts = 0
    for i in range(1):
        # 读取向量
        results  = read_file(os.path.join(model_save_path, "vec_{}.pkl".format(i)))
        vec_pool = list()
        for fos in results:
            # 低频次的fos_L3被过滤掉
            if fos not in filtered_fos:
                continue
            for year in results[fos]:
                for vec in results[fos][year]:
                    vec_pool.append((fos, vec))
        del results
        # 
        random.shuffle(vec_pool)
        start_idx = 0
        end_idx   = 0
        batch_num = math.ceil(len(vec_pool) / batch_size)
        for j in range(batch_num):
            # 将读取的向量划分为batch输入encoder和decoder
            end_idx    = min(start_idx + batch_size, len(vec_pool))
            batch_data = vec_pool[start_idx: end_idx]
            batch_enc_in  = list() # vector
            batch_dec_in  = list() # [SOS] + fos
            batch_dec_out = list() # fos + [EOS]
            for fos, vec in batch_data:
                batch_enc_in.append(vec)
                batch_dec_in.append("[SOS] " + fos)
                batch_dec_out.append(fos + " [EOS]")
            batch_enc_in       = torch.tensor(batch_enc_in, dtype=torch.float32).to(device)
            input_for_in       = tokenizer(batch_dec_in,  
                                           padding='max_length', 
                                           max_length=max_length_in,
                                           truncation=True, 
                                           return_tensors='pt', 
                                           add_special_tokens=False)
            input_ids_in       = torch.tensor(input_for_in['input_ids']).to(device)
            attention_mask_in  = torch.tensor(input_for_in['attention_mask']).to(device)
            # token_type_ids_in  = torch.tensor(input_for_in['token_type_ids']).to(device)
            input_for_out      = tokenizer(batch_dec_out, 
                                           padding='max_length', 
                                           max_length=max_length_in, 
                                           truncation=True, 
                                           return_tensors='pt', 
                                           add_special_tokens=False)
            input_ids_out      = torch.tensor(input_for_out['input_ids']).to(device)
            attention_mask_out = torch.tensor(input_for_out['attention_mask']).to(device)
            # token_type_ids_out = torch.tensor(input_for_out['token_type_ids']).to(device)
            
            # ENCODER
            mu, log_sigma_squared = encoder(batch_enc_in)
            
            # discontinus (简单的autoencoder)
            z = mu
            
            # continus (vae)
            # e = torch.empty_like(mu).normal_(mean=0., std=1.)
            # z = mu + e * torch.sqrt(torch.exp(log_sigma_squared))
            
            # DECODER
            SOS_token = tokenizer.encode("[SOS]")[1]
            EOS_token = tokenizer.encode("[EOS]")[1]
            batch_decoded_words = list()
            smaples, _ = z.shape
            for s in range(smaples):
                decoder_input  = torch.tensor([[SOS_token]], device=device)  # SOS   
                decoder_hidden = z[None, s:s+1, :]
                decoded_words  = list()
                for di in range(max_length_in):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                        decoded_words.append(EOS_token)
                        break
                    else:
                        decoded_words.append(topi.item())
                    decoder_input = topi.squeeze().detach()
                    decoder_input = decoder_input.view(1, 1)
                batch_decoded_words.append(tokenizer.decode(decoded_words))
                
            # 判断预测精度
            right_count = 0
            batch_count = 0
            for fos_i, fos_j in zip(batch_dec_out, batch_decoded_words):
                batch_count += 1
                if fos_i == fos_j:
                    right_count += 1
            right_counts += right_count
            batch_counts += batch_count
            print("File {} Bacth {} {:.4f}".format(i, j, right_counts / batch_counts))
            
            # 清空缓存
            batch_enc_in  = list() # vector
            batch_dec_in  = list() # [SOS] + fos
            batch_dec_out = list() # fos + [EOS]
            torch.cuda.empty_cache()
            start_idx  = end_idx
    