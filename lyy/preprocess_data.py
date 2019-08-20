# coding: utf-8
import pandas as pd
from scipy import sparse
import os
import gc
import math
import numpy as np
import json
import random

def gen10_22day_log():
    f_path = '../data/'

    df_static = pd.read_csv('../data/map_ad_static.out', sep='\t', 
              names=['aid','create_time','advertiser','good_id','good_type',
                     'ad_type_id','ad_size']).sort_values(by='create_time')

    opt_log = pd.read_csv('../data/final_map_bid_opt.out',
                          sep='\t',names=['aid','time',
                        'opt_type','ad_type','cost_type','bib']).sort_values(by='time')


    for day in range(13):
        print(day)

        f_name = 'track_log_201904' + str(10+day) + '.out'

        totalExpos = pd.read_csv(f_path+f_name, sep='\t',
                                 names=['ad_request_id','ad_request_time',
                                 'user_id','wei_id','ad_info'])


        out = {}
        for i in range(totalExpos.shape[0]):
            wei_i = str(totalExpos.loc[i,'wei_id'])
            tp_log_i = totalExpos.loc[i,'ad_info']
            str_log_i = tp_log_i.split(';')
            for ad_j in str_log_i:
                str_ad_j = ad_j.split(',')
                ad_i_key = str(str_ad_j[0])
                if  ad_i_key not in out:
                    try:
                        inf = df_static.loc[df_static['aid'] == int(ad_i_key)].values[0]
                        advertiser = int(inf[2])
                        good_id = int(inf[3])
                        good_type = int(inf[4])
                        ad_type_id = int(inf[5])
                    except:
                        advertiser = -1
                        good_id = -1
                        good_type = -1
                        ad_type_id = -1

                    try:
                        opt_tr = opt_log.loc[opt_log['aid'] == int(ad_i_key)]
                        index_opt = opt_tr['time'].idxmax()
                        ad_type = int(opt_tr.loc[index_opt,'ad_type'])
                        cost_type = int(opt_tr.loc[index_opt,'cost_type'])
                    except:
                        ad_type = -1
                        cost_type = -1

                    tmp_dic = {'total_eps':0,'total_rqt':0,'wei_eps1':0,'wei_rqt1':0,
                               'wei_eps2':0,'wei_rqt2':0,'wei_eps3':0,'wei_rqt3':0,
                               'wei_eps4':0,'wei_rqt4':0,'advertiser':advertiser,
                              'good_id':good_id,'good_type':good_type,'ad_type_id':ad_type_id,
                              'ad_type':ad_type,'cost_type':cost_type}
                    if str_ad_j[6] == '1':
                        tmp_dic['total_rqt'] = tmp_dic['total_rqt'] + 1
                        tmp_dic['total_eps'] = tmp_dic['total_eps'] + 1
                        tmp_dic['wei_rqt'+wei_i] = tmp_dic['wei_rqt'+wei_i] + 1
                        tmp_dic['wei_eps'+wei_i] = tmp_dic['wei_eps'+wei_i] + 1
                        out[ad_i_key] = tmp_dic
                    else:
                        tmp_dic['total_rqt'] = tmp_dic['total_rqt'] + 1
                        tmp_dic['wei_rqt'+wei_i] = tmp_dic['wei_rqt'+wei_i] + 1
                        out[ad_i_key] = tmp_dic
                else:
                    if str_ad_j[6] == '1':
                        out[ad_i_key]['total_rqt'] = out[ad_i_key]['total_rqt'] + 1
                        out[ad_i_key]['total_eps'] = out[ad_i_key]['total_eps'] + 1
                        out[ad_i_key]['wei_rqt'+wei_i] = out[ad_i_key]['wei_rqt'+wei_i] + 1
                        out[ad_i_key]['wei_eps'+wei_i] = out[ad_i_key]['wei_eps'+wei_i] + 1
                    else:
                        out[ad_i_key]['total_rqt'] = out[ad_i_key]['total_rqt'] + 1
                        out[ad_i_key]['wei_rqt'+wei_i] = out[ad_i_key]['wei_rqt'+wei_i] + 1
        out_path = './cache/day' + str(10+day) + '.json'
        f = open(out_path,'w',encoding = 'utf-8')
        json.dump(out,f,indent = 4,sort_keys = True)
        f.close()
#################################################################################  
def gen23day_log():
    
    totalExpos = pd.read_csv('../data/test_tracklog_20190423.last.out', sep='\t',
                             names=['ad_request_id','ad_request_time',
                             'user_id','wei_id','ad_info'])

    df_static = pd.read_csv('../data/map_ad_static.out', sep='\t', 
              names=['aid','create_time','advertiser','good_id','good_type',
                     'ad_type_id','ad_size']).sort_values(by='create_time')

    opt_log = pd.read_csv('../data/final_map_bid_opt.out',
                          sep='\t',names=['aid','time',
                        'opt_type','ad_type','cost_type','bib']).sort_values(by='time')
    out = {}
    for i in range(totalExpos.shape[0]):
        wei_i = str(totalExpos.loc[i,'wei_id'])
        tp_log_i = totalExpos.loc[i,'ad_info']
        str_log_i = tp_log_i.split(';')
        flag = 1
        for ad_j in str_log_i:
            str_ad_j = ad_j.split(',')
            ad_i_key = str(str_ad_j[0])
            if  ad_i_key not in out:
                try:
                    inf = df_static.loc[df_static['aid'] == int(ad_i_key)].values[0]
                    advertiser = int(inf[2])
                    good_id = int(inf[3])
                    good_type = int(inf[4])
                    ad_type_id = int(inf[5])
                except:
                    advertiser = -1
                    good_id = -1
                    good_type = -1
                    ad_type_id = -1

                try:
                    opt_tr = opt_log.loc[opt_log['aid'] == int(ad_i_key)]
                    index_opt = opt_tr['time'].idxmax()
                    ad_type = int(opt_tr.loc[index_opt,'ad_type'])
                    cost_type = int(opt_tr.loc[index_opt,'cost_type'])
                except:
                    ad_type = -1
                    cost_type = -1

                tmp_dic = {'total_eps':0,'total_rqt':0,'wei_eps1':0,'wei_rqt1':0,
                           'wei_eps2':0,'wei_rqt2':0,'wei_eps3':0,'wei_rqt3':0,
                           'wei_eps4':0,'wei_rqt4':0,'advertiser':advertiser,
                          'good_id':good_id,'good_type':good_type,'ad_type_id':ad_type_id,
                          'ad_type':ad_type,'cost_type':cost_type}
                if str_ad_j[5] == '0' and flag == 1:
                    flag = 0
                    tmp_dic['total_rqt'] = tmp_dic['total_rqt'] + 1
                    tmp_dic['total_eps'] = tmp_dic['total_eps'] + 1
                    tmp_dic['wei_rqt'+wei_i] = tmp_dic['wei_rqt'+wei_i] + 1
                    tmp_dic['wei_eps'+wei_i] = tmp_dic['wei_eps'+wei_i] + 1
                    out[ad_i_key] = tmp_dic
                else:
                    tmp_dic['total_rqt'] = tmp_dic['total_rqt'] + 1
                    tmp_dic['wei_rqt'+wei_i] = tmp_dic['wei_rqt'+wei_i] + 1
                    out[ad_i_key] = tmp_dic
            else:
                if str_ad_j[5] == '0' and flag == 1:
                    flag = 0
                    out[ad_i_key]['total_rqt'] = out[ad_i_key]['total_rqt'] + 1
                    out[ad_i_key]['total_eps'] = out[ad_i_key]['total_eps'] + 1
                    out[ad_i_key]['wei_rqt'+wei_i] = out[ad_i_key]['wei_rqt'+wei_i] + 1
                    out[ad_i_key]['wei_eps'+wei_i] = out[ad_i_key]['wei_eps'+wei_i] + 1
                else:
                    out[ad_i_key]['total_rqt'] = out[ad_i_key]['total_rqt'] + 1
                    out[ad_i_key]['wei_rqt'+wei_i] = out[ad_i_key]['wei_rqt'+wei_i] + 1
    out_path = './cache/day' + str(23) + '.json'
    f = open(out_path,'w',encoding = 'utf-8')
    json.dump(out,f,indent = 4,sort_keys = True)
    f.close()
    
####################################################################
def gen_prob_log_b23():

    f_path = '../data/'
    test = pd.read_csv(f_path+'Btest_select_request_20190424.out',sep='\t',
                         names=['aid','request_fifo'])
    load_dic = []
    for j in range(14):
        out_path = './cache/day' + str(10+j) + '.json'
        with open(out_path,'r') as load_f:
            load_dic.append(json.load(load_f))

    cnt = 0
    ad_list = []
    for i in range(test.shape[0]):
        ad_i = str(test.loc[i,'aid'])
        for j in range(14):
             if ad_i in load_dic[j]:
                 ad_list.append(ad_i)
                 cnt = cnt + 1
                 break
    lis_out = []
    for i in ad_list:
        tmp = {'aid':i}
        for j in range(14):
            if i in load_dic[j]:
                tmp['ad_type'] = load_dic[j][i]['ad_type']
                tmp['ad_type_id'] = load_dic[j][i]['ad_type_id']
                tmp['advertiser'] = load_dic[j][i]['advertiser']
                tmp['cost_type'] = load_dic[j][i]['cost_type']
                tmp['good_id'] = load_dic[j][i]['good_id']
                tmp['good_type'] = load_dic[j][i]['good_type']
                tmp['day'+str(10+j)+'total_eps'] = load_dic[j][i]['total_eps']
                tmp['day'+str(10+j)+'total_rqt'] = load_dic[j][i]['total_rqt']
                tmp['day'+str(10+j)+'wei_eps1'] = load_dic[j][i]['wei_eps1']
                tmp['day'+str(10+j)+'wei_eps2'] = load_dic[j][i]['wei_eps2']
                tmp['day'+str(10+j)+'wei_eps3'] = load_dic[j][i]['wei_eps3']
                tmp['day'+str(10+j)+'wei_eps4'] = load_dic[j][i]['wei_eps4']
                tmp['day'+str(10+j)+'wei_rqt1'] = load_dic[j][i]['wei_rqt1']
                tmp['day'+str(10+j)+'wei_rqt2'] = load_dic[j][i]['wei_rqt2']
                tmp['day'+str(10+j)+'wei_rqt3'] = load_dic[j][i]['wei_rqt3']
                tmp['day'+str(10+j)+'wei_rqt4'] = load_dic[j][i]['wei_rqt4']
            else:
                tmp['day'+str(10+j)+'total_eps'] = 0
                tmp['day'+str(10+j)+'total_rqt'] = 0
                tmp['day'+str(10+j)+'wei_eps1'] = 0
                tmp['day'+str(10+j)+'wei_eps2'] = 0
                tmp['day'+str(10+j)+'wei_eps3'] = 0
                tmp['day'+str(10+j)+'wei_eps4'] = 0
                tmp['day'+str(10+j)+'wei_rqt1'] = 0
                tmp['day'+str(10+j)+'wei_rqt2'] = 0
                tmp['day'+str(10+j)+'wei_rqt3'] = 0
                tmp['day'+str(10+j)+'wei_rqt4'] = 0
        lis_out.append(tmp)

    df = pd.DataFrame(lis_out)
    df.to_csv('./cache/prob_log_b23.csv',encoding = 'utf-8',index = False)
##########################################################################
def gen_test_prob():
    test = pd.read_csv('../data/Btest_select_request_20190424.out',sep='\t',
                   names=['ad_id','request_fifo'])

    test_sample_bid = pd.read_csv('../data/Btest_sample_bid.out',sep='\t',
                              names=['id','ad_id','type1','type2','bib'])
    
    load_dic = []
    for j in range(14):
        out_path = './cache/day' + str(10+j) + '.json'
        with open(out_path,'r') as load_f:
            load_dic.append(json.load(load_f))

    ad_list22 = []
    ad_list23 = []
    for i in range(test.shape[0]):
        ad_i = str(test.loc[i,'ad_id'])
        if ad_i in load_dic[13] and load_dic[13][ad_i]['total_rqt'] > 300:
            ad_list23.append(ad_i)
        elif ad_i in load_dic[12]:
            ad_list22.append(ad_i)
    print(len(ad_list23),len(ad_list22))
    ##########################################
    out_list22 = {}
    w = []
    A_ = 0.71
    B_ = -0.36
    C_ = -0.134
    for k in range(11,0,-1):
        y = 1/(A_*k + B_) + C_
        #y = 32/(25*k - 17.5)-0.08
        #y = 0.001*4.6**(12-k)
        w.append(y)
    for aid in ad_list22:
        tmp = {'total_eps':0,'total_rqt':0,'wei_eps1':0,'wei_eps2':0,'wei_eps3':0,'wei_eps4':0,
              'wei_rqt1':0,'wei_rqt2':0,'wei_rqt3':0,'wei_rqt4':0}
        
        old_eps_wei = [0,0,0,0]
        old_rqt_wei = [0,0,0,0] 
    
        for day in range(2,13,1):
            if aid in load_dic[day]:
                tmp['ad_type'] = load_dic[day][aid]['ad_type']
                tmp['ad_type_id'] = load_dic[day][aid]['ad_type_id']
                tmp['advertiser'] = load_dic[day][aid]['advertiser']
                tmp['cost_type'] = load_dic[day][aid]['cost_type']
                tmp['good_id'] = load_dic[day][aid]['good_id']
                tmp['good_type'] = load_dic[day][aid]['good_type']
                tmp['total_eps'] = load_dic[day][aid]['total_eps']*w[day-2] + tmp['total_eps']
                tmp['total_rqt'] = load_dic[day][aid]['total_rqt']*w[day-2] + tmp['total_rqt']
                
                for wei_index in range(1,5,1):
                    tmp['wei_eps'+str(wei_index)] = tmp['wei_eps'+str(wei_index)]+                    load_dic[day][aid]['wei_eps'+str(wei_index)]*w[day-2]
                    
                    tmp['wei_rqt'+str(wei_index)] = tmp['wei_rqt'+str(wei_index)]+                    load_dic[day][aid]['wei_rqt'+str(wei_index)]*w[day-2]
                    
        out_list22[aid] = tmp
        
    #####################################
    out_list23 = {}
    A_ = 0.66
    B_ = -0.525
    C_ = -0.125
    w = []
    for k in range(12,0,-1):
        y = 1/(A_*k + B_) + C_
        #y = 32/(25*k - 17.5)-0.08
        #y = 0.001*4.6**(12-k)
        w.append(round(y,3))
    print(w)
    for aid in ad_list23:
        tmp = {'total_eps':0,'total_rqt':0,'wei_eps1':0,'wei_eps2':0,'wei_eps3':0,'wei_eps4':0,
              'wei_rqt1':0,'wei_rqt2':0,'wei_rqt3':0,'wei_rqt4':0}
        
        old_eps_wei = [0,0,0,0]
        old_rqt_wei = [0,0,0,0] 
    
        for day in range(2,14,1):
            if aid in load_dic[day]:
                tmp['ad_type'] = load_dic[day][aid]['ad_type']
                tmp['ad_type_id'] = load_dic[day][aid]['ad_type_id']
                tmp['advertiser'] = load_dic[day][aid]['advertiser']
                tmp['cost_type'] = load_dic[day][aid]['cost_type']
                tmp['good_id'] = load_dic[day][aid]['good_id']
                tmp['good_type'] = load_dic[day][aid]['good_type']
                tmp['total_eps'] = load_dic[day][aid]['total_eps']*w[day-2] + tmp['total_eps']
                tmp['total_rqt'] = load_dic[day][aid]['total_rqt']*w[day-2] + tmp['total_rqt']
                
                for wei_index in range(1,5,1):
                    tmp['wei_eps'+str(wei_index)] = tmp['wei_eps'+str(wei_index)]+                    load_dic[day][aid]['wei_eps'+str(wei_index)]*w[day-2]
                    
                    tmp['wei_rqt'+str(wei_index)] = tmp['wei_rqt'+str(wei_index)]+                    load_dic[day][aid]['wei_rqt'+str(wei_index)]*w[day-2]
                    
        out_list23[aid] = tmp
        
    f22 = open('./cache/test22.json','w',encoding = 'utf-8')
    json.dump(out_list22,f22,indent = 4,sort_keys = True)
    f22.close()  
 
    f23 = open('./cache/test23.json','w',encoding = 'utf-8')
    json.dump(out_list23,f23,indent = 4,sort_keys = True)
    f23.close() 
#################################################################
###主程序
file_name = 'cache'
# 创建保中间结果的文件夹
if not os.path.exists('./%s' % file_name):
    os.makedirs('./%s' % file_name)
gen10_22day_log()
gen23day_log()
gen_test_prob()
