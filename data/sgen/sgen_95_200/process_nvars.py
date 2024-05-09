import os
import pickle
import numpy as np

def get_num_literal(instance):
    #print(type(instance))
    flat_list = np.asarray([item for sublist in instance for item in sublist])
    sorted_literals = np.unique(np.absolute(flat_list))
    return len(sorted_literals)


directory = os.fsencode("/home/galen/Haonan/NeuroSAT/data/sgen/sgen_95_200")
nvars =[]
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    #label = []
    instance = []
    if filename.endswith("cnf"):
        #print(label)
        #with open(filename) as f:
        with open(filename) as file:
            lines = file.readlines()
            for line in lines:
                clause = line.strip().split()
                clause = [int(x) for x in clause]
                instance.append(clause[:-1])
            nvars += [get_num_literal(instance)]
label_file=open("nvars.pkl", "wb")
pickle.dump(nvars, label_file)
#print(nvars)
