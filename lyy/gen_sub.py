import pandas as pd
from scipy import sparse
import os
import gc
import math
import numpy as np
import json
import random

def ronghe():
    sub1 = pd.read_csv('../submission/gdy_lgb.csv',header=None,names=['sample_id','pred1'])
    sub2 = pd.read_csv('../submission/wh_lgb.csv',header=None,names=['sample_id','pred2'])
    sub3 = pd.read_csv('../submission/guize.csv',header=None,names=['sample_id','pred3'])
    sub4 = pd.read_csv('../submission/gdy_NN.csv',header=None,names=['sample_id','pred4'])

    sub1['pred2'] = sub2['pred2'].values  #wh_lgb
    sub1['pred3'] = sub3['pred3'].values  #gz
    sub1['pred4'] = sub4['pred4'].values  #gdy_NN

    sub1['pred'] = None
    #新广告 由规则结果提取
    sub1.loc[sub1.pred3 == -1 , 'pred'] = sub1.loc[sub1.pred3 == -1 , 'pred1']
    #旧广告 由 ( ((gdy_lgb**0.8)*(gdy_nn**0.2)) ** 0.5 ) * (wh_lgb**0.5) 
    sub1.loc[sub1.pred3 == 1 , 'pred'] = (((sub1.loc[sub1.pred3 == 1,'pred1']**0.8)*(sub1.loc[sub1.pred3==1,'pred4']**0.2))**0.5)*(sub1.loc[sub1.pred3==1,'pred2']**0.5)
    
    
    sub1[['sample_id','pred']].to_csv('./ronghesub.csv',header=None,index=False)
    
    
f_path = '../data/'
with open('./cache/test22.json','r') as load_f:
    test22= json.load(load_f)

with open('./cache/test23.json','r') as load_f:
    test23 = json.load(load_f)
    
test = pd.read_csv(f_path+'Btest_select_request_20190424.out',sep='\t',
                   names=['ad_id','request_fifo'])

test_sample_bid = pd.read_csv(f_path+'Btest_sample_bid.out',sep='\t',
                              names=['id','ad_id','type1','type2','bib'])

opt_log = pd.read_csv(f_path+'final_map_bid_opt.out',sep='\t',
                      names=['ad_id','time','opt_type','ad_type','cost_type','bib'])



cont = dict(test_sample_bid['ad_id'].value_counts())
#############################################################
out_new_old = []

for i in range(62968):
    tmp = []
    tmp.append(i+1)
    out_new_old.append(tmp)

print(len(out_new_old))
for i in cont:
    l = cont[i]
    tr = test_sample_bid.loc[test_sample_bid['ad_id']==i]
    if str(i) not in test22 and str(i) not in test23 :
        for j in range(l):
            index = tr['bib'].idxmin()
            out_new_old[index].append(-1)
            tr = tr.drop(index)
    else:
        for j in range(l):
            index = tr['bib'].idxmin()
            out_new_old[index].append(1)
            tr = tr.drop(index)
            
df = pd.DataFrame(out_new_old)
df.to_csv('../submission/guize.csv',encoding = 'utf-8',index = False,header = None)
ronghe()
nnsub = pd.read_csv('ronghesub.csv',names=['id','expose'])
#################################################################
out = []

for i in range(62968):
    tmp = []
    tmp.append(i+1)
    out.append(tmp)

for i in cont:
    l = cont[i]
    if str(i) not in test22 and str(i) not in test23 :
        tr = test_sample_bid.loc[test_sample_bid['ad_id']==i]
        request_fifo = test.loc[test['ad_id'] == i]
        request_index = request_fifo['ad_id'].idxmin()
        s = request_fifo.loc[request_index,'request_fifo']
        s1 = s.split('|')
        new_num = len(s1)
        smooth = 0.001
            
        for j in range(l):
            index = tr['bib'].idxmin()
            eps = nnsub.loc[index,'expose']
            #eps_lgb = lgbsub.ix[index,'expose']
            #eps = (eps**0.6)*(eps_lgb**0.4)
            #eps = 1
            if eps > new_num:
                eps = new_num
            eps = round(eps)
            eps = eps + smooth
            smooth = smooth + 0.001
            out[index].append(eps)
            tr = tr.drop(index)
    else:
        tr = test_sample_bid.loc[test_sample_bid['ad_id']==i]
        request_fifo = test.loc[test['ad_id'] == i]
        request_index = request_fifo['ad_id'].idxmin()
        s = request_fifo.loc[request_index,'request_fifo']
        s1 = s.split('|')
        wei_li = [0,0,0,0]
        for q_i in s1:
            q_i2 = q_i.split(',')
            if q_i2[1] == '1':
                wei_li[0] = wei_li[0] + 1
            elif q_i2[1] == '2':
                wei_li[1] = wei_li[1] + 1
            elif q_i2[1] == '3':
                wei_li[2] = wei_li[2] + 1
            elif q_i2[1] == '4':
                wei_li[3] = wei_li[3] + 1
                
        if str(i) in test23 :
            eps1 = 0
            for wei in range(4):
                if test23[str(i)]['wei_rqt'+str(wei+1)] == 0:
                    eps1 = eps1 +                     wei_li[wei]*test23[str(i)]['total_eps']/test23[str(i)]['total_rqt']
                else:
                    eps1 = eps1 +                     wei_li[wei]*test23[str(i)]['wei_eps'+str(wei+1)]/                    test23[str(i)]['wei_rqt'+str(wei+1)]
        else:     
            eps1 = 0
            for wei in range(4):
                if test22[str(i)]['wei_rqt'+str(wei+1)] == 0:
                    eps1 = eps1 +                     wei_li[wei]*test22[str(i)]['total_eps']/test22[str(i)]['total_rqt']
                else:
                    eps1 = eps1 +                     wei_li[wei]*test22[str(i)]['wei_eps'+str(wei+1)]/                    test22[str(i)]['wei_rqt'+str(wei+1)]
            
        index = tr['bib'].idxmin()
        eps_nn = nnsub.loc[index,'expose']
        ad_type = tr.loc[index,'type1']
        if ad_type == 10:
            eps1 = (eps1**0.2)*(eps_nn**0.8)
        elif ad_type == 0:
            eps1 = eps1
        elif ad_type == 13:
            eps1 = (eps1**0.4)*(eps_nn**0.6)
        elif ad_type == 5:
            eps1 = (eps1**0.6)*(eps_nn**0.4)
        else:
            eps1 = (eps1**0.4)*(eps_nn**0.6)
        smooth = 0.001
        opt_tr = opt_log.loc[opt_log['ad_id'] == i]
###################
        for j in range(l):
            index = tr['bib'].idxmin()
            if opt_tr.shape[0] == 0:
            #if 1:
                eps = eps1
                #eps = round(eps)
            else:
                index_opt = opt_tr['time'].idxmax()
                old_bib = opt_tr.loc[index_opt,'bib']
                new_bib = tr.loc[index,'bib']
                eps = eps1*new_bib/old_bib
                #eps = eps1
            eps = round(eps)
            eps = eps + smooth
            smooth = smooth + 0.001
            out[index].append(eps)
            tr = tr.drop(index)
            
df = pd.DataFrame(out)
df.to_csv('../submission.csv',encoding = 'utf-8',index = False,header = None)
