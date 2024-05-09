import os
import pickle
import numpy as np

def get_num_literal(instance):
    #print(type(instance))
    flat_list = np.asarray([item for sublist in instance for item in sublist])
    sorted_literals = np.unique(np.absolute(flat_list))
    return len(sorted_literals)

def remove_space(instance):
    sorted_literals = np.unique(np.absolute(np.asarray([item for sublist in instance for item in sublist])))
    num_literals = len(sorted_literals)
    #print(sorted_literals)
    #print(instance)
    if sorted_literals[-1] == num_literals:
        return instance
        #print(sorted_literals)
    #print(np.argwhere(sorted_literals != np.arange(1, num_literals+1)))
    wrong_index = np.argwhere(sorted_literals != np.arange(1, num_literals+1))[0]
    num_wrong_literal = sorted_literals[wrong_index]
    for i in range(len(instance)):
        for j in range(len(instance[i])):
            if abs(instance[i][j]) < num_wrong_literal:
                continue
            else:
                correct_id = list(sorted_literals).index(abs(instance[i][j])) + 1
                if instance[i][j] > 0:
                    instance[i][j] = correct_id
                else:
                    instance[i][j] = -correct_id
        #:wq
    #sorted_literals = np.unique(np.absolute(np.asarray([item for sublist in instance for item in sublist])))
    #print(sorted_literals)
    #print(" ")
        #num_literals = len(sorted_literals)
    return instance


directory = "/home/galen/Haonan/NeuroSAT/data/sgen/sgen_15_10000"
nvars =[]
for i in range(10000):
    filename = os.path.join(directory, "sgen-"+str(i)+".pkl")
    #label = []
    #instance = []
    #if filename.endswith("pkl"):
        #print(label)
        #with open(filename) as f:
        #with open(filename) as file:
        #    lines = file.readlines()
        #    for line in lines:
        #       if not line.startswith("p"):
        #            clause = line.strip().split()
        #            clause = [int(x) for x in clause]
        #            instance.append(clause[:-1])
    with open(filename, 'rb') as f:
        lines = pickle.load(f)
    nvars += [get_num_literal(lines)]
label_file=open("nvars.pkl", "wb")
pickle.dump(nvars, label_file)
#print(nvars)
