#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from os import path
import gc
import time
import numpy as np
from tqdm import tqdm
from dask import compute, delayed
import dask
try:
   import cPickle as pickle
except:
   import pickle

#from dask_mpi import initialize
#initialize()


def merge(d1, d2):
        for item in d2:
                if item in d1.keys():
                        d1[item] += d2[item]
                else:
                        d1[item] = d2[item]

def L_func(cols,col_count_ser,n):
    l1= np.sqrt(np.ones(len(cols)) * col_count_ser[cols]/N ) * np.log(n/col_count_ser[cols])
    l2= np.sqrt(np.ones(len(cols)) * col_count_ser[cols]/N ) * np.log(n/col_count_ser[cols])
    return np.dot(l1,l2)

@dask.delayed
def make_pairs(k):
    Sim_pair_k=dict()
    v=dict_A[k]
    for (k2,v2) in dict_A.items():
        #if ((k,k2) not in Sim_pair.keys()) and ((k,k2) not in Sim_pair.keys()):
        common_keys= set(v).intersection(set(v2))
        #Sim_pair_k[(k,k2)] = L_func(common_keys,col_count_ser,n)
        Sim_pair_k[k2] = L_func(common_keys,col_count_ser,n)
        #else:
        #    continue
    with open("./Sim_pair/Sim_pair_"+str(i)+".pickle","wb") as pickle_out:
        pickle.dump(Sim_pair_k, pickle_out)
    return {k:pd.Series(Sim_pair_k)}

if __name__ == '__main__':

    data = "./clustering_Biden/"

    N= 10000000

    #n_workers = int(os.getenv('NSLOTS'))
    #print("n_workers: ", n_workers)
    #from dask.distributed import LocalCluster
    #cluster = LocalCluster()
    
    from dask_jobqueue import SGECluster
    cluster = SGECluster(cores=28, memory='200GB', job_extra=['-P scv', '-pe mpi_28_tasks_per_node 112'])
    #cluster.scale(4) 

    from dask.distributed import Client, progress
    #client = Client("tcp://192.168.19.195:8786")
    client = Client(cluster)
    print(client)

    entries=os.listdir(data)
    
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

    #num_cores=os.getenv('NSLOTS')
    #print("num_cores: ", num_cores) 
    Sim_pair=dict()
    inputs = dict_A.keys()
    
    #cluster = LocalCluster(n_workers=num_cores, dashboard_address=0)


    #with parallel_backend('dask'):
    #    Parallel(verbose=100)(delayed(make_pairs)(i) for i in inputs)   
    #for i in inputs:
    Sim_pair_k = client.map(make_pairs, inputs)
    #Sim_pair = Sim_pair_k

    Sim_pair = client.gather(Sim_pair_k)
    print(len(Sim_pair))
    Sim_pair_df = pd.DataFrame(Sim_pair)
    
    
    with open("./Sim_pair.pickle","wb") as pickle_out:
        pickle.dump(Sim_pair, pickle_out)
    client.close() 
