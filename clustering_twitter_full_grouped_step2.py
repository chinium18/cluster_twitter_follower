#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from os import path
import gc
import time
import numpy as np
#from tqdm import tqdm
from multiprocessing import Pool,Manager
import multiprocessing 
from joblib import Parallel, delayed
try:
   import cPickle as pickle
except:
   import pickle
import argparse
import glob
from tqdm import tqdm

def merge(d1, d2):
        for item in d2:
                if item in d1.keys():
                        d1[item] += d2[item]
                else:
                        d1[item] = d2[item]


def L_func(cols,r,r2,n):
    cols_N = cols & N_keys
    N_list=np.float_([N[i] for i in cols_N])
    
    l1= np.sqrt(r*np.ones(len(cols_N)) * col_count_ser[cols_N]/N_list ) * np.log((n+1)/(col_count_ser[cols_N]+1))
    l2= np.sqrt(r2*np.ones(len(cols_N)) * col_count_ser[cols_N]/N_list ) * np.log((n+1)/(col_count_ser[cols_N]+1))
    return np.dot(l1,l2)


def make_pairs(k):
    v=dict_A[k]
    r=row_count[k]
    for (k2,v2) in dict_A.items():
        common_keys= set(v).intersection(set(v2))
            #print(len(common_keys))
        r2=row_count[k2]
        Sim_pair[(k,k2)] = L_func(common_keys,r,r2,n)
    return Sim_pair

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="starting pos and chunck size")
    parser.add_argument('index',type=int)
    parser.add_argument('chunk_size',type=int)
    
    args = parser.parse_args()
    index = int(args.index)
    chunk_size = int(args.chunk_size)
    print("index: ",index)
    print("chunk_size: ",chunk_size)
#    data = "./clustering_Biden/"
    data = "./mentioner_blocks/"

    #N= 10000000

    entries=glob.glob(data+"Biden*.csv")
    col_count = dict()
    with open('./mentioner_blocks/big6_feature_follower_2410166_June9.pickle', 'rb') as bfile:
        N = pickle.load(bfile)
    
    with open('./mentioner_blocks/selected_features_sum_June10.pickle', 'rb') as cfile:
        col_count = pickle.load(cfile)

    with open('mentioner_blocks/mentioner_read_June6_check.pickle','rb') as dfile:
        mentioner = set(pickle.load(dfile))
    #from multiprocessing import Process, Lock, Manager
    # About 1min/file, may need parallization later.
    if (not path.exists("./dict_A.pickle")):
        print("Please run step 1 first!")
        exit;
        #    pickle.dump(col_count, pickle_col)
    else:
        with open("./dict_A.pickle","rb") as afile:
            dict_A = pickle.load(afile)    
        with open("./row_count.pickle","rb") as rfile:
            row_count = pickle.load(rfile)    
        #with open("./col_count.pickle","rb") as cfile:
        #    col_count = pickle.load(cfile)    
    print("N size is ", len(N)) 
    print(len(dict_A))
    print(len(row_count))
    #print(len(col_count))
    
    N_keys = set(N.keys()) 
    col_count_ser = pd.Series(col_count).astype(int)
    #row_count_ser = pd.Series(row_count)
    
    
    n=sum(col_count.values())
    print(n)


    num_cores=multiprocessing.cpu_count()
    print("num_cores: ", num_cores) 

    t1 = time.time()
    key_list = list(dict_A.keys())
#    for k in key_list[rank*chunk:(rank+1)*chunk]:
#        make_pairs(k, Sim_pair)

    #processed_list = Parallel(n_jobs=num_cores)(delayed(make_pairs)(i) for i in inputs)
    
    Sim_pair=dict()
    with Pool(num_cores) as p:
        inputs = key_list
        Sim_pair_list = list(p.map(make_pairs, inputs[index*chunk_size:(index+1)*chunk_size]))

    t2 = time.time()
    print("time used: ",t2-t1)

    Sim_pair_all=dict()
    print(len(Sim_pair_list))
    
    Sim_pair_all_df = pd.DataFrame(np.zeros((chunk_size, len(key_list))),columns=key_list, index = key_list[index*chunk_size:(index+1)*chunk_size])
    for i in tqdm(Sim_pair_list): 
    #    Sim_pair_all.update(i)
        for key in i:
            Sim_pair_all_df.loc[key[0], key[1]] = i[key]
    
    Sim_pair_all_df.to_parquet("./Sim_pair_mp/Sim_pair_"+str(index)+"_test.parquet") 
   # with open("./Sim_pair_mp/Sim_pair_"+str(index)+".pickle","wb") as pickle_out:
   #     pickle.dump(Sim_pair_all, pickle_out)
    
