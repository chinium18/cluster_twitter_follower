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
#from multiprocessing import Pool,Manager
import multiprocessing 
from joblib import Parallel, delayed
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


def L_func(cols,col_count_ser,n):
    l1= np.sqrt(np.ones(len(cols)) * col_count_ser[cols]/N ) * np.log(n/col_count_ser[cols])
    l2= np.sqrt(np.ones(len(cols)) * col_count_ser[cols]/N ) * np.log(n/col_count_ser[cols])
    return np.dot(l1,l2)


def make_pairs(k, Sim_pair):
    v=dict_A[k]
    for (k2,v2) in od_A.items():
        if ((k,k2) not in Sim_pair.keys()) and ((k,k2) not in Sim_pair.keys()):
            common_keys= set(v).intersection(set(v2))
            #print(len(common_keys))
            Sim_pair[(k,k2)] = L_func(common_keys,col_count_ser,n)
        else:
            continue

# In[5]:

if __name__ == '__main__':

    data = "./clustering_Biden/"

    N= 10000000



# In[6]:


    entries=os.listdir(data)
    
    
    # In[7]:
    
    
    #from multiprocessing import Process, Lock, Manager
    # About 1min/file, may need parallization later.
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
    from collections import OrderedDict
    
    
    #row_count_ser = pd.Series(row_count)
    
    


   # m = Manager() 

   # Sim_pair=m.dict()
    num_cores=os.getenv('NSLOTS')
    print("num_cores: ", num_cores) 

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    Sim_pair=dict()
    chunk = len(row_count) // int(num_cores)
    print("chunk: ",chunk)
    #od_A = OrderedDict(sorted(dict_A.items(), key=lambda t:t[0])) 
    #col_count_ser = pd.Series(col_count)
    if rank == 0:
        od_A = OrderedDict(sorted(dict_A.items(), key=lambda t:t[0]))
        row_count = row_count
        col_count = col_count
        col_count_ser = pd.Series(col_count)
        print(len(od_A))
        print(type(od_A))
        print(len(row_count))
        print(len(col_count))
        n=sum(col_count.values())
        print(n)
    else:
        od_A = None
        row_count = None
        col_count = None
        col_count_ser = None
        n = None
    od_A = comm.bcast(od_A, root = 0)
    row_count = comm.bcast(row_count, root = 0)
    col_count = comm.bcast(col_count, root = 0)
    col_count_ser = comm.bcast(col_count_ser, root = 0)
    n = comm.bcast(n, root = 0) 
#   inputs = tqdm(dict_A.keys())
    
    key_list = list(od_A.keys())
    for k in key_list[rank*chunk:(rank+1)*chunk]:
        make_pairs(k, Sim_pair) 
    #processed_list = Parallel(n_jobs=num_cores)(delayed(make_pairs)(i) for i in inputs)
    
#    with Pool(pool_size) as p:
#        p.map(make_pairs, dict_A.keys())

#for (k,v) in tqdm(dict_A.items()):
    Sim_pair = comm.gather(Sim_pair, root=0)
    if rank == 0:
        Sim_pair_all = dict()
        for i in range(len(Sim_pair)):
            Sim_pair_all.update(Sim_pair[i])
        print(len(Sim_pair_all))
        with open("./Sim_pair.pickle","wb") as pickle_out:
            pickle.dump(Sim_pair_all, pickle_out)
    else:
        assert Sim_pair is None
    
    comm.Barrier()
    MPI.Finalize() 
