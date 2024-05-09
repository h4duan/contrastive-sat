import os
import pickle

directory = os.fsencode("/home/galen/Haonan/NeuroSAT/data/sgen/sgen_95_200")
label =[]
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    #label = []
    if filename.endswith("label"):
        #print(label)
        with open(filename) as f:
            first_line = f.readline().strip()
            #print(first_line)
            label += [int(first_line)]
            #print(label)
            #print(" ")
label_file=open("label.pkl", "wb")
pickle.dump(label, label_file)

