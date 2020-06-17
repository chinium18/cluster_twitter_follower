try:
   import cPickle as pickle
except:
   import pickle


with open("./Sim_pair/Sim_pair_@BarbaraDarlin.pickle","rb") as afile:
    user = pickle.load(afile)


print(user)


