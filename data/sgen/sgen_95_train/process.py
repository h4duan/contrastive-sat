import os
import pickle

directory = os.fsencode("/home/galen/Haonan/NeuroSAT/data/sgen/sgen_95_train")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("cnf"):
        instance = []
        with open(filename) as file:
            lines = file.readlines()
            for line in lines:
                clause = line.strip().split()
                clause = [int(x) for x in clause]
                instance.append(clause[:-1])
        new_filename = filename[:-3] + "pkl"
        out_file = open(new_filename, "wb")
        pickle.dump(instance, out_file)
        #print(instance)

