''''
'''
from numpy import *
import matplotlib.pyplot as plt
import sys

"This is used to load a file created by rfe"
def load_rfe(filename):
    fo = open(filename)
    data = fo.readlines()
    SSE = []
    features = []
    for i in range(len(data)):
        if i%2==0:
            SSE.append(float(data[i]))
        else:
            temp = data[i].split(",")
            for i in range(len(temp)):
                temp[i] = int(temp[i])
            features.append(temp)
    fo.close()
    return SSE, features

def min_indecies(weights, features, discard):
    my_list = []
    for i in range(weights.shape[0]):
        my_list.append((features[i], abs(weights[i])))

    my_list.sort(key=lambda x: x[1])
    my_list = my_list[discard:]

    features = []
    for tup in my_list:
        features.append(tup[0])

    features = sorted(features)
    return features


def load_data(filename, c, feature_load = False):
    '''
    Loads data from _filname_.
    Input:
    - str filename: name of a file to load from.
    Output:
    - ndarray Y: a vector (size n) of true classes.
    - numpy_matrix X: a nxd matrix of gene expression data.
    - ndarray L: a vector (size d) of feature labels.
    '''
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

    #Adding Ys to the X.
    Y = (ones(len(label))*c).reshape((len(label),1))
    data = hstack((Y,X))
    
    return data, sample_names, feature_names

def load_data2(filename):
    # Load a csv of floats:
    ret = genfromtxt(filename, delimiter=",")
    return ret

#concatinates data into a single matrix and separating out test data set.
#I ended up not using this at all.
def processData(datalist, test):
    print
    print "Shuffling data..."
    print "Separating test and training sets..."
    
    output = datalist[0]
    for i in range(1, len(datalist)):
        output = vstack((output, datalist[i]))
    random.shuffle(output)
    train = output[:output.shape[0]-test,:]
    test = output[output.shape[0]-test:,:]
    return train, test

#testing purpose, loads data from assignment 1 
def load_test():
    #loads data
    training_data = genfromtxt('hw2-data/train.txt', delimiter=',')
    Y_train = training_data[:,0]
    X_train = training_data[:, 1:]
    Y_test = genfromtxt('hw2-data/test_label.txt', delimiter=',')
    X_test = genfromtxt('hw2-data/test.txt', delimiter=',')
    
    return Y_train, X_train, Y_test, X_test

def find_min(x, y):
    mindex = 0
    minerror = float("inf")
    for i in range(len(x)):
        if y[i] < minerror:
            mindex = i
            minerror = y[i]
    return x[mindex]
    
def loadlog(filename, line):
    fo = open(filename)
    lines = fo.readlines()
    c = lines[line].split(",")
    for i in range(len(c)):
        c[i] = float(c[i])
    fo.close()
    return c

'''
The functions below are just used to check the validity of data.
Just testing purpose 
'''
#I used this function to verfiy if the examples are staying valid after shuffling
def stamp(Y,X):
    stamp = zeros(X.shape[1])
    for i in range(Y.shape[0]):
        if Y[i] == 1:
            stamp = stamp + X[i,:].T
    return stamp.sum()

#checks if given lists are the same, checking if all data have the same features.
def isSame(lists):
    for l in lists[1:]:
        for i in range(len(lists[0])):
            if lists[0][i] != l[i]:
                print lists[0][i], l[i]
                return False
    return True
