# importing necessary modules. 
from numpy import *
#from logistic_regression import *
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model

def load_data(filename, c, feature_load = False):
    print
    print "Loading " + filename + "..."
    
    #Be careful rows are features and columns are samples.
    #Only reading the first line as label
    with open(filename, 'r') as f:
        label = f.readline()
    label = label.split(",")
    label[-1] = label[-1][:-1]

    print "This data contains " + str(len(label)) + " samples"
    sample_names = asarray(label)
    
    #Loading the actual data
    feature_names = None
    if(feature_load):
        inputdata = genfromtxt(filename, delimiter=',', skip_header=1, dtype = str)
        feature_names = inputdata[:,0]
    
    #Separating out the feature labels
    inputdata = genfromtxt(filename, delimiter=',', skip_header=1, dtype = float)
    print "This data contains " + str(inputdata.shape[0]) + " features"
    X = inputdata[:, 1:]

    #Transposing the X matrix because i like that way:
    #The world is better when samples are rows and features are columns.
    X = X.T
    
    return X, feature_names


def transform(X, c):
    num = X.shape[0]
    newX = None
    for i in range(len(c)):
        if c[i] != 0:
            if newX == None:
                newX = X[:,i].reshape((num,1))
            else:
                newX = hstack((newX,X[:,i].reshape(num,1)))
    return newX

def data_w_selected_features():
    data_0, feature_names = load_data("MILES_data/amlMiles2012.csv", 0, True)
    data_1, dmy = load_data("MILES_data/cALLMiles2014.csv", 1)
    data_2, dmy = load_data("MILES_data/CLLMiles2014.csv", 2)
    data_3, dmy = load_data("MILES_data/mdsMiles2014.csv", 3)

    folder = "RFE_output_4subtypes/"
    fo = open(folder+"RFE_union_features.txt")
    lines = fo.readlines()
    c = lines[0].split(",")
    for i in range(len(c)):
        c[i] = int(float(c[i]))
    fo.close()
    
    savetxt("selected_mile/RFE/aml.csv", transform(data_0,c), delimiter=",")
    savetxt("selected_mile/RFE/cALL.csv", transform(data_1,c), delimiter=",")
    savetxt("selected_mile/RFE/CLL.csv", transform(data_2,c), delimiter=",")
    savetxt("selected_mile/RFE/mds.csv", transform(data_3,c), delimiter=",")

    features=[]
    for i in range(len(c)):
        if c[i] != 0:
            features.append(feature_names[i])
   
    f1 = open("selected_mile/RFE/features.csv", "w")
    f1.write(str(features)[1:-1])
    f1.close()

def find_redundant_features():
    data_0, feature_names = load_data("MILES_data/amlMiles2012.csv", 0, True)

    filename = "RFE_output_4subtypes/RFE_union_features.txt"
    fo = open(filename)
    lines = fo.readlines()
    RFE = lines[0].split(",")
    for i in range(len(RFE)):
        RFE[i] = int(float(RFE[i]))
    fo.close()

    filename = "LASSO_output_4subtypes/LASSO_union_features.txt"
    fo = open(filename)
    lines = fo.readlines()
    LASSO = lines[0].split(",")
    for i in range(len(LASSO)):
        LASSO[i] = int(float(LASSO[i]))
    fo.close()

    filename = "VO_output_4subtypes/VO_union_features.txt"
    fo = open(filename)
    lines = fo.readlines()
    VO = lines[0].split(",")
    for i in range(len(VO)):
        VO[i] = int(float(VO[i]))
    fo.close()

    features = []
    for i in range(len(RFE)):
        if RFE[i] > 0 and LASSO[i] > 0 and VO[i] > 0:
            features.append(i)

    for feature in features:
        print feature_names[feature]

def main():
    find_redundant_features()
    #data_w_selected_features()
    
if __name__ == "__main__":
    main()
