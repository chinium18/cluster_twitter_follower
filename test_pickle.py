import pickle
import os
import glob
directory = "./mentioner_blocks/Biden*.csv"
for fn in glob.glob(directory):
    print(fn)

#with open('./mentioner_blocks/big6_feature_follower_2410166_June9.pickle', 'rb') as afile:
#    dict_A = pickle.load(afile)
#print(dict_A)
#with open('./mentioner_blocks/selected_features_sum_June10.pickle', 'rb') as bfile:
#    dict_B = pickle.load(bfile)
#print(dict_B)
with open('mentioner_blocks/mentioner_read_June6_check.pickle','rb') as cfile:
    dict_C = pickle.load(cfile)
print(dict_C)

