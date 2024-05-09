import os
import pickle

directory = "/home/galen/Haonan/NeuroSAT/data/sgen/sgen_15_10000"
label =[]
for i in range(10000):
    filename = os.path.join(directory, "sgen-"+str(i)+".label")
    #label = []
    #if filename.endswith("label"):
        #print(label)
    with open(filename) as f:
        first_line = f.readline().strip()
            #print(first_line)
    label += [int(first_line)]
            #print(label)
            #print(" ")
label_file=open("label.pkl", "wb")
pickle.dump(label, label_file)

