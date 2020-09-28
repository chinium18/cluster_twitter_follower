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

if __name__ == '__main__':
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
        t1 = time.time()
        # A dictionary that holds all records in A
        dict_A = dict()
        # A dictionary that holds row counts
        row_count = dict()
        # A dictionary that holds row counts
        
        
        for file in entries:
        #def reformat(dict_A, row_count, col_count, file):
            #file = data+file
            #print(file)
            df = pd.read_csv(file, index_col="Author")
            #print(df.head(1))
            # Update column sum and row sum
            #merge(col_count,df.count().to_dict())
            row_count.update(df.sum(axis=1).to_dict())
            #print(len(col_count))
            #print(len(row_count))
            for Author in (set(df.index) & mentioner):
                cols=df.columns[df.loc[Author,:].notnull()]
                dict_A[Author] = cols.tolist()
                #print(dict_A)
            #print(df.info())    
            del df
            gc.collect()
        
        
        t2 = time.time()
        print(t2-t1)
        
        
        with open("./dict_A.pickle","wb") as pickle_dict_A:
            pickle.dump(dict_A, pickle_dict_A)
        with open("./row_count.pickle","wb") as pickle_row:
            pickle.dump(row_count, pickle_row)
        #with open("./col_count.pickle","wb") as pickle_col:
        #    pickle.dump(col_count, pickle_col)
    else:
        with open("./dict_A.pickle","rb") as afile:
            dict_A = pickle.load(afile)    
        with open("./row_count.pickle","rb") as rfile:
            row_count = pickle.load(rfile)    
        #with open("./col_count.pickle","rb") as cfile:
        #    col_count = pickle.load(cfile)    
    
    print(len(dict_A))
    print(len(row_count))
    print(len(col_count))
    
