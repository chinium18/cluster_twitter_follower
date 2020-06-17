#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from os import path
import gc
import time
import numpy as np
from tqdm import tqdm
#import pickle
from multiprocessing import Pool,Manager
import numba as nb

try:
   import cPickle as pickle
except:
   import pickle
# In[2]:



# In[3]:


def merge(d1, d2):
        for item in d2:
                if item in d1.keys():
                        d1[item] += d2[item]
                else:
                        d1[item] = d2[item]


# In[4]:
@nb.jit(nopython=True)
def L_func(cols,r,r2,n):
    l1= np.sqrt(r*np.ones(len(cols)) * col_count_ser[cols]/N ) * np.log((n+1)/(col_count_ser[cols]+1))
    l2= np.sqrt(r2*np.ones(len(cols)) * col_count_ser[cols]/N ) * np.log((n+1)/(col_count_ser[cols]+1))
    return np.dot(l1,l2)


@nb.jit(nopython=True)
def make_pairs(k,k2):
    v=dict_A[k]
    v2=dict_A[k2]
    r=row_count[k]
    if ((k,k2) not in Sim_pair.keys()) and ((k,k2) not in Sim_pair.keys()):
        common_keys= set(v).intersection(set(v2))
        r2=row_count[k2]
        #print(len(common_keys))
        Sim_pair[(k,k2)] = L_func(common_keys,r,r2,n)
    #return Sim_pair

'''
@nb.jit(nopython=True)
def make_pairs(k, k2):
    v=dict_A[k]
    v2=dict_A[k2]
    common_keys= set(v).intersection(set(v2))
    #print(len(common_keys))
    Sim_pair[(k,k2)] = L_func(common_keys,col_count_ser,n)
'''
# In[5]:

if __name__ == '__main__':

    data = "./clustering_Biden/"

    N= 10000000

    entries=os.listdir(data)
    
    if (not path.exists("./dict_A.pickle")):
        t1 = time.time()
        # A dictionary that holds all records in A
        dict_A = dict()
        # A dictionary that holds row counts
        row_count = dict()
        # A dictionary that holds row counts
        col_count = dict()
        
        
        for file in tqdm(entries):
        #def reformat(dict_A, row_count, col_count, file):
            file = data+file
            #print(file)
            df = pd.read_csv(file, index_col="Author")
            #print(df.head(1))
            # Update column sum and row sum
            merge(col_count,df.count().to_dict())
            row_count.update(df.sum(axis=1).to_dict())
            #print(len(col_count))
            #print(len(row_count))
            for Author in df.index:
                cols=df.columns[df.loc[Author,:].notnull()]
                dict_A[Author] = cols.tolist()
                #print(dict_A)
            #print(df.info())    
            del df
            gc.collect()
        
        
        t2 = time.time()
        print(t2-t1)
        
        
        # In[9]:
        
        
        with open("./dict_A.pickle","wb") as pickle_dict_A:
            pickle.dump(dict_A, pickle_dict_A)
        with open("./row_count.pickle","wb") as pickle_row:
            pickle.dump(row_count, pickle_row)
        with open("./col_count.pickle","wb") as pickle_col:
            pickle.dump(col_count, pickle_col)
    else:
        with open("./dict_A.pickle","rb") as afile:
            dict_A = pickle.load(afile)    
        with open("./row_count.pickle","rb") as rfile:
            row_count = pickle.load(rfile)    
        with open("./col_count.pickle","rb") as cfile:
            col_count = pickle.load(cfile)    
    
    print(len(dict_A))
    print(len(row_count))
    print(len(col_count))
    
    
    col_count_ser = pd.Series(col_count)
    #row_count_ser = pd.Series(row_count)
    
    
    n=sum(col_count.values())
    print(n)
   
    from numba import types
    from numba.typed import Dict
    Dict_A = Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type[:])
    for k, v in dict_A.items():
        Dict_A[k] = [v]

   # m = Manager() 
    @nb.jit(nopython=True)
    def outer_pair():
        Sim_pair=Dict()
        for i in Dict_A.keys():
            for j in Dict_A.keys(): 
                make_pairs(i, j)
        return Sim_pair

    Sim_pair = outer_pair()
    print(len(Sim_pair))
    
    
    with open("./Sim_pair.pickle","wb") as pickle_out:
        pickle.dump(Sim_pair, pickle_out)
    
