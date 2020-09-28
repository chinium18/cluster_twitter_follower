try:
   import cPickle as pickle
except:
   import pickle
import pandas as pd

with open("./test.pickle","rb") as afile:
	dict_A0= pickle.load(afile)

#with open('./mentioner_blocks/selected_features_sum_June10.pickle', 'rb') as afile:
#	a = pickle.load(afile)

with open('./mentioner_blocks/mentioner_read_June6_check.pickle', 'rb') as bfile:
        mentioner = pickle.load(bfile)


batch_size = 2153

df = pd.DataFrame(np.zeros((batch_size, len(mentioner))), columns = mentioner)




#print(user)
#n = list(N.keys())
#m = n[:4]

#mn = set(n) & set(m)

#print(mn)


