import dask.array as da

x = da.random.random((100000,100000), chunks=(1000,1000))

x[x<0.95] = 0

import sparse
s = x.map_blocks(sparse.COO)

s.sum(axis=0)[:100].compute()

s.todense()
