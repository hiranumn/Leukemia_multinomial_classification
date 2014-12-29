'''
CSE 546 Machine Learning
Final Project: Multiclass regression for the subtypes of leukemia

Professor Carlos Guestrin
Project advisor: Harley Montgomery

This code conducts multi-class regression on MILE data, which contains gene expression data for leukemia patients, and classifies them in to various subtypes of leukemia. 

Things to do.
1. Lets see if I can do 1v1 classification on this problem ;)


Acknowledgement:
I would like to thank professor Su-In Lee for helping with the project idea.
'''

# importing necessary modules. 
from numpy import *
from logistic_regression import *
from softmax_regression import *
from feature_selection import *
from svm import *
from util import *
import matplotlib.pyplot as plt
import sys

'''
Conduct multinomial regression using an all-pairwise approach. This is an approach similar to the one taken by the paper "Clinical Utility of Microarray-Based Gene Expression Profiling in the Diagnosis and Subclassification of Leukemia: Report From the International Microarray Innovations in Leukemia Study Group". For every pair of 2 classes, I will train a binary classifier. Given n classes, this will result in n^2/2 classifiers that classifies every possible calss pair (i,j). Let us call this H_i_j.
The actuall classification process works in the following way: for every data point x, I will run x throw every single classifier H_i_j. Every time x runs through a binary classifier, a vote P(Y=i|x) is added to class i and a vote P(Y=j|x) is added to class j. After running through all binary classifiers, x is classfied in to a class with the highest vote value.

In the paper, top 100 features were selected form each classifiers, and union of those were used in the actual classification process.
This is not implemented yet.
'''
def all_pairwise_approach(mode, names, datafiles):
    train_ratio = 0.8 #what proportion of data you want to use as a test set.

    #loading data
    test_d_list = []
    train_d_list = []

    #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
    for i in range(len(datafiles)):
        data = load_data2(datafiles[i])
        train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
        test_d_list.append(data[int(train_ratio*data.shape[0]):,:]) #test data

    #initializing classifiers
    if mode == "logistic_regression":
        print "Conducting logistic regression..."
        classifiers = []
        for i in range(len(datafiles)):
            for j in range(i+1, len(datafiles)):
                lr = logistic_regressor(0.3, 0.01, label_0=names[i], label_1=names[j])
                #train data
                X = vstack((train_d_list[i],train_d_list[j]))
                Y = hstack((zeros(train_d_list[i].shape[0]),ones(train_d_list[j].shape[0])))
                lr.load(Y,X)
                lr.change_termination_condition(threshold = 0.0000005)
                lr.bgd_l2(False)
                classifiers.append(lr)
            
        print "all classifiers trained..."

        results = []
        #testing them on test data set
        for i in range(len(datafiles)):
            result = zeros(test_d_list[i].shape[0]*len(datafiles)).reshape((test_d_list[i].shape[0],len(datafiles)))
            for classifier in classifiers:
                prob = classifier.make_predictions(test_d_list[i])
                for j in range(prob.shape[0]):
                    result[j, names.index(classifier.label_1)] += prob[j]
                    result[j, names.index(classifier.label_0)] += 1-prob[j]
            results.append(result)

        print "all predictions are made..."
    elif mode == "svm":
        print "using svm to classify leukemia cancers"
        classifiers = []
        for i in range(len(datafiles)):
            for j in range(i+1, len(datafiles)):
                X = vstack((train_d_list[i],train_d_list[j]))
                Y = hstack((-1*ones(train_d_list[i].shape[0]),ones(train_d_list[j].shape[0])))
                svm = SVM(Y, X, run=200, eta=0.01, label_0=names[i], label_1=names[j])
                #train data
                svm._sgd()
                classifiers.append(svm)

        print "all classifiers trained..."

        results = []
        #testing them on test data set
        for i in range(len(datafiles)):
            result = zeros(test_d_list[i].shape[0]*len(datafiles)).reshape((test_d_list[i].shape[0],len(datafiles)))
            for classifier in classifiers:
                pred = classifier.make_predictions(test_d_list[i])
                for j in range(pred.shape[0]):
                    if pred[j] < 0:
                        result[j, names.index(classifier.label_0)] += 1.0
                    elif pred[j] > 0:
                        result[j, names.index(classifier.label_1)] += 1.0
                    else:
                        result[j, names.index(classifier.label_0)] += .5
                        result[j, names.index(classifier.label_1)] += .5
                        
            results.append(result)

        print "all predictions are made..."
        
    num_correct_total = 0
    num_wrong_total = 0
    for i in range(len(datafiles)):
        num_correct = 0
        num_wrong = 0
        for j in range(results[i].shape[0]):
            maxindex = 0
            maxprob = 0
            for t in range(len(datafiles)):
                if results[i][j,t] > maxprob:
                    maxindex = t
                    maxprob = results[i][j,t]
            if maxindex == i:
                num_correct_total += 1
                num_correct += 1
            else:
                num_wrong_total += 1
                num_wrong += 1
        print names[i], num_correct*1.0/(num_correct+num_wrong)

    print num_correct_total
    print num_wrong_total
    print num_correct_total*1.0/(num_correct_total+num_wrong_total)
                    
'''
This algorithm is adapted by Calros's lecture slides.
This basically constructs n independent binary classifiers by making one class a pivot.
This is the same as softmax approach
'''
def pivot_approach():
    train_ratio = 0.8 #what proportion of data you want to use as a test set.

    #here are the datafiles to load from
    names = ["aml", "cALL", "CLL", "mds"]
    datafiles = ["processed_mile/aml.csv","processed_mile/cALL.csv","processed_mile/CLL.csv","processed_mile/mds.csv"]

    #loading data
    test_d_list = []
    train_d_list = []

    #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
    for i in range(len(datafiles)):
        data = load_data2(datafiles[i])
        train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
        test_d_list.append(data[int(train_ratio*data.shape[0]):,:]) #test data

    #initializing classifiers
    classifiers = []
    for i in range(1, len(datafiles)):
        #instantiate logistic regressor
        lr = logistic_regressor(0.0, 0.01, label_0=names[0], label_1=names[i])
        
        #train data
        X = vstack((train_d_list[0],train_d_list[i]))
        Y = hstack((zeros(train_d_list[0].shape[0]),ones(train_d_list[i].shape[0])))
        lr.load(Y,X)
        lr.change_termination_condition(threshold = 0.0000005)
        lr.bgd_l2(False)
        print "testing 1", (lr.make_predictions(test_d_list[0])).sum()/test_d_list[0].shape[0]
        #print "testing 2", (lr.make_predictions(test_d_list[i])).sum()/test_d_list[i].shape[0]
        classifiers.append(lr)
            
    print "all classifiers trained..."
    
    results = []
    #testing them on test data set
    for i in range(len(datafiles)):
        result = zeros(test_d_list[i].shape[0]*len(datafiles)).reshape((test_d_list[i].shape[0],len(datafiles)))

        #make prediction for the pivot
        denom = ones(test_d_list[i].shape[0])
        for classifier in classifiers:
            denom += exp(test_d_list[i].dot(classifier.w)+classifier.w_0)

        for j in range(result.shape[0]):
            result[j,0] = 1.0/denom[j]

        counter = 1
        for classifier in classifiers:
            prediction = exp(test_d_list[i].dot(classifier.w)+classifier.w_0)/denom
            for j in range(result.shape[0]):
                result[j,counter] = prediction[j]
            counter += 1
            
        results.append(result)

    print "all predictions made..."
    
    num_correct_total = 0
    num_wrong_total = 0
    for i in range(len(datafiles)):
        num_correct = 0
        num_wrong = 0
        for j in range(results[i].shape[0]):
            maxindex = 0
            maxprob = 0
            for t in range(len(datafiles)):
                if results[i][j,t] > maxprob:
                    maxindex = t
                    maxprob = results[i][j,t]
            if maxindex == i:
                num_correct_total += 1
                num_correct += 1
            else:
                num_wrong_total += 1
                num_wrong += 1
        print names[i], num_correct*1.0/(num_correct+num_wrong)

    print num_correct_total
    print num_wrong_total
    print num_correct_total*1.0/(num_correct_total+num_wrong_total)

#Instead of 1v1 all pairwise, it conducts 1 vs rest using pre-selected featrues.
def one_versus_rest_approach(mode, names, datafiles):
    train_ratio = 0.8 #what proportion of data you want to use as a test set.

    #loading data
    test_d_list = []
    train_d_list = []

    #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
    for i in range(len(datafiles)):
        data = load_data2(datafiles[i])
        train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
        test_d_list.append(data[int(train_ratio*data.shape[0]):,:]) #test data

    #initializing classifiers
    #logistic regression version
    if mode == "logistic_regression":
        print "Conducting logistic regression..."
        classifiers = []
        for i in range(len(datafiles)):
            lr = logistic_regressor(0.3, 0.01, label_0=names[i], label_1="rest")
            #train data
            #stack everything for X.
            X = None
            for j in range(len(datafiles)):
                if X == None:
                    X = train_d_list[j]
                else:
                    X = vstack((X,train_d_list[j]))
                    
            #generate Y label
            Y = None
            for j in range(len(datafiles)):
                if Y == None:
                    if i==j:
                        Y = zeros(train_d_list[j].shape[0])
                    else:
                        Y = ones(train_d_list[j].shape[0])
                else:
                    if i==j:
                        Y = hstack((Y,zeros(train_d_list[j].shape[0])))
                    else:
                        Y = hstack((Y,ones(train_d_list[j].shape[0])))
                        
            lr.load(Y,X)
            lr.change_termination_condition(threshold = 0.0000005)
            lr.bgd_l2(False)
            classifiers.append(lr)
            
        print "all classifiers trained..."

        results = []
        
        #testing them on test data set
        for i in range(len(datafiles)):
            result = zeros(test_d_list[i].shape[0]*len(datafiles)).reshape((test_d_list[i].shape[0],len(datafiles)))
            for classifier in classifiers:
                prob = classifier.make_predictions(test_d_list[i])
                for j in range(prob.shape[0]):
                    result[j, names.index(classifier.label_0)] += 1-prob[j]
            results.append(result)

        print "all predictions are made..."

    #svm version
    elif mode == "svm":
        print "using svm to classify leukemia cancers"
        classifiers = []
        for i in range(len(datafiles)):
            #stack everything for X.
            X = None
            for j in range(len(datafiles)):
                if X == None:
                    X = train_d_list[j]
                else:
                    X = vstack((X,train_d_list[j]))
                    
            #generate Y label
            Y = None
            for j in range(len(datafiles)):
                if Y == None:
                    if i==j:
                        Y = -1*ones(train_d_list[j].shape[0])
                    else:
                        Y = ones(train_d_list[j].shape[0])
                else:
                    if i==j:
                        Y = hstack((Y,-1*ones(train_d_list[j].shape[0])))
                    else:
                        Y = hstack((Y,ones(train_d_list[j].shape[0])))
                        
            svm = SVM(Y, X, run=300, eta=0.01, label_0=names[i], label_1="rest")
            svm._sgd()
            classifiers.append(svm)

        print "all classifiers trained..."

        results = []
        #testing them on test data set
        for i in range(len(datafiles)):
            result = zeros(test_d_list[i].shape[0]*len(datafiles)).reshape((test_d_list[i].shape[0],len(datafiles)))
            for classifier in classifiers:
                prob = classifier.make_predictions(test_d_list[i])
                for j in range(prob.shape[0]):
                    result[j, names.index(classifier.label_0)] += -1*prob[j]
            results.append(result)

        print "all predictions are made..."
        
    num_correct_total = 0
    num_wrong_total = 0
    for i in range(len(datafiles)):
        num_correct = 0
        num_wrong = 0
        for j in range(results[i].shape[0]):
            maxindex = 0
            maxprob = 0
            for t in range(len(datafiles)):
                if results[i][j,t] > maxprob:
                    maxindex = t
                    maxprob = results[i][j,t]
            if maxindex == i:
                num_correct_total += 1
                num_correct += 1
            else:
                num_wrong_total += 1
                num_wrong += 1
        print names[i], num_correct*1.0/(num_correct+num_wrong)

    print num_correct_total
    print num_wrong_total
    print num_correct_total*1.0/(num_correct_total+num_wrong_total)

def softmax_approach(names, datafiles):
    train_ratio = 0.8 #what proportion of data you want to use as a test set.


    #loading data
    test_d_list = []
    train_d_list = []

    #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
    for i in range(len(datafiles)):
        data = load_data2(datafiles[i])
        train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
        test_d_list.append(data[int(train_ratio*data.shape[0]):,:]) #test data
    
    X = None
    for j in range(len(datafiles)):
        if X == None:
            X = train_d_list[j]
        else:
            X = vstack((X,train_d_list[j]))
                    
    #generate Y label
    Y = None
    for j in range(len(datafiles)):
        if Y == None:
            Y = zeros(train_d_list[j].shape[0])
        else:
            Y = hstack((Y,j*ones(train_d_list[j].shape[0])))

    #train a classifer
    sr = softmax_regressor(0.3, 0.01, labels=names)
    sr.load(Y,X)
    sr.change_termination_condition(threshold = 0.0000005)
    sr.bgd(False)

    results = []
    for i in range(len(datafiles)):
        prob = sr.make_predictions(test_d_list[i])
        results.append(prob)

    num_correct_total = 0
    num_wrong_total = 0
    for i in range(len(datafiles)):
        num_correct = 0
        num_wrong = 0
        for j in range(results[i].shape[0]):
            maxindex = 0
            maxprob = 0
            temp = results[i][j,:]
            for t in range(len(temp)):
                if temp[t] > maxprob:
                    maxindex = t
                    maxprob = temp[t]
            if maxindex == i:
                num_correct_total += 1
                num_correct += 1
            else:
                num_wrong_total += 1
                num_wrong += 1
        print names[i], num_correct*1.0/(num_correct+num_wrong)

    print num_correct_total
    print num_wrong_total
    print num_correct_total*1.0/(num_correct_total+num_wrong_total)

def main():
    print 
    print "Nao's Final Project :)"
    print "Using the MILE (Microarray Innovations in Leukemia) data"

    
    #here are the datafiles to load from
    names = ["ALL_with_hyperdiploid_karyotype", "ALL_with_t(1;19)", "ALL_with_t(12;21)", "AML_complex_aberrant_karyotype", "AML_with_inv(16)_t(16;16)","AML_with_normal_karyotype_+_other_abnormalities","AML_with_t(8;21)","AML_with_t(11q23)_MLL","AML_with_t(15;17)","c-ALL_Pre-B-ALL_with_t(9;22)","c-ALL_Pre-B-ALL_without_t(9;22)","CLL","CML","mature_B-ALL_with_t(8;14)","MDS","Non-leukemia_and_healthy_bone_marrow","Pro-B-ALL_with_t(11q23)_MLL","T-ALL"]
    folder = "MILES_data_18subtypes/"
    datafiles = []
    for i in range(len(names)):
        datafiles.append(folder+names[i]+".csv")

    '''
    a list of multinomial regression you can conduct
    '''
    '''
    print "APA-svm"
    all_pairwise_approach("svm", names, datafiles)
    
    print "APA-lr"
    all_pairwise_approach("logistic_regression", names, datafiles)

    print "OVR-svm"
    one_versus_rest_approach("svm", names, datafiles)

    print "OVR-lr"
    one_versus_rest_approach("logistic_regression", names, datafiles)

    print "Softmax"
    softmax_approach(names, datafiles)
    '''
    '''
    feature selection stuff
    '''
    vo = Variable_ordering(names, datafiles)
    vo.run()

    #rfe = Recursive_feature_elimination(names, datafiles)
    #rfe.run()
    
if __name__ == "__main__":
    main()
