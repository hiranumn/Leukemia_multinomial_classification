# importing necessary modules. 
from numpy import *
import matplotlib.pyplot as plt
import sys
import csv
def load():
    print "Starting dataload"
    filename = "data_18/tab_delimited_MILEs.tsv"

    test = open(filename, 'r')
    data = test.readlines()

    features = []

    #processing the first line
    ID = (data[0].split("\t"))[2:]
    ID[-1] = ID[-1][:-1]

    #process the rest
    X = []
    features = []
    
    for i in range(1,len(data)):
        temp = data[i].split("\t")
        for j in range(2,len(temp)):
            temp[j] = float(temp[j])
        features.append(temp[:2])
        X.append(temp[2:])
        
    X = array(X)
    features = array(features)
    
    return ID, X, features

def load_class():
    print "Starting classload"
    filename = "data_18/GSE13159.txt"

    test = open(filename, 'r')
    data = test.readlines()
    classified = []

    for i in range(1,len(data)):
        temp = data[i].split("\t")
        temp[2] = temp[2][:-1]
        classified.append(temp)

    return classified

def extract_class(c):
    clas = []
    for sample in c:
        if not sample[2] in clas:
            clas.append(sample[2])
    return clas

def main():
    ID, X, features = load()
    c = load_class()
    
    c_list = extract_class(c)
    
    cluster = [[] for x in xrange(len(c_list))]

    for sample in c:
        target = c_list.index(sample[2])
        data = ID.index(sample[0])
        cluster[target].append(X[:,data].reshape((X.shape[0],1)))

    for i in range(len(cluster)):
        output = None
        for item in cluster[i]:
            if output == None:
                output = item
            else:
                output = hstack((output, item))
        output = output.T
        raw_name = (c_list[i]).split(" ")
        name = ""
        for i in raw_name:
            name = name + i + "_"
            
        raw_name = name.split("/")
        name = ""
        for i in raw_name:
            name = name + i + "_"
        
        savetxt("MILES_2/"+name[:-2]+".csv", output, delimiter=",")
            
            
        
        
            
    

    
    
            



if __name__ == "__main__":
    main()
