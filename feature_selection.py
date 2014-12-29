
from numpy import *
from logistic_regression import *
from softmax_regression import *
from svm import *
from util import *
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model
import scipy as sp


class Recursive_feature_elimination:
    def __init__(self, names, datafiles, elim_rate = 0.1, err_rate_th = 0.05):
        self.elim_rate = elim_rate
        self.err_rate_th = err_rate_th
        self.names = names
        self.datafiles = datafiles


    ##runs multiple crossvalidation to determine a set of features for all binary classifiers
    ##returns a union of the selected featrues for later use.
    def run(self):
        train_ratio = 0.8 #what proportion of data you want to use as a test set.

        #here are the datafiles to load from
        #loading data
        test_d_list = []
        train_d_list = []

        #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
        for i in range(len(self.datafiles)):
            #change this load function depending on the type of file you are reading from
            #data, dmy, dmy = load_data(self.datafiles[i], i, True)
            data = genfromtxt(self.datafiles[i], delimiter=",")
            train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
            
        for i in range(len(self.datafiles)):
            for j in range(i+1, len(self.datafiles)):
                Y = hstack((-1*ones(train_d_list[i].shape[0]),ones(train_d_list[j].shape[0])))
                X = vstack((train_d_list[i][:,1:],train_d_list[j][:,1:]))
                self.run_one_svm_cv(Y,X, self.names[i], self.names[j], fold=5)
        
        
    def run_one_svm_cv(self, Y, X, label_0, label_1, fold=None):
        if fold == None: #conducting loocv.
            print "LOOCV not yet implemented"
            print "fold needs to be a interger bigger than 0"
            return None
        else:
            print "Doing " + label_0 + "vs" + label_1
            cur_f = 1
            f_min = 0.0001 #runs to 2 features.
            
            #features to use
            cur_features = range(X.shape[1])
            old_features = range(X.shape[1])
        
            #separating data into k blocks
            Y_blocks = []
            X_blocks = []
            size = Y.shape[0]
            for i in range(fold):
                Y_sub = Y[size*i/fold:size*(i+1)/fold]
                X_sub = X[size*i/fold:size*(i+1)/fold,:]
                Y_blocks.append(Y_sub)
                X_blocks.append(X_sub)

            log = []

            #do this until current c hits c min, which is arbitrarily set.
            prev_error = 1

            #at least use f_min of the features
            while f_min<cur_f:
                error = 0
                ##creating test and validation sets
                for i in range(fold):
                    #picking out validation set.
                    test_Y = Y_blocks[i]
                    test_X = X_blocks[i]
                    train_Y = None
                    train_X = None

                    #assembling other folds into training set.
                    for j in range(fold):
                        if j != i:
                            if train_Y == None:
                                train_Y = Y_blocks[j]
                                train_X = X_blocks[j]
                            else:
                                train_Y = hstack((train_Y, Y_blocks[j]))
                                train_X = vstack((train_X, X_blocks[j]))

                    #start training svm using selected features
                    svm = SVM(train_Y, train_X[:,cur_features], run=50, eta=0.01)
                    svm._sgd()
                    error += svm.calculate_errors(test_Y, test_X[:,cur_features])

                error = error/fold
                print "error :", error, " #features: ", len(cur_features)

                log.append([error,cur_features])

                #now reduce features by magnitude:
                cur_f = cur_f*(1-self.elim_rate)
                cur_features = min_indecies(svm.w, cur_features, int(1-cur_f*X.shape[1]))
                
        #save result
        if save:
            output = ""
            for entry in log:
                output += str(entry[0]) +"\n"
                output += str(entry[1])[1:-1] +"\n"
            f1 = open("RFE_output/"+ label_0+"_vs_"+label_1+".txt", "w")
            f1.write(output)
            f1.close()
        return log

    def get_union_features(self):
        folder = "RFE_output/"
        num_features = 17726
        union_features = [0]*num_features
        for i in range(len(self.names)):
            for j in range(i+1, len(self.names)):
                dmy, features = load_rfe(folder+self.names[i]+"_vs_"+self.names[j]+".txt")
                feature = features[-25] #this number is arbitrarily set
                for t in feature:
                    union_features[t] = union_features[t] + 1
                    
        f1 = open("RFE_output/RFE_union_features.txt", "w")
        f1.write(str(union_features)[1:-1])
        f1.close()

        count = [0,0,0,0,0,0,0]
        for i in union_features:
            count[int(i)]+=1
        print count

#similar to RFE orders values with cosign between vectors.
class Variable_ordering:
    def __init__(self, names, datafiles):
        self.test = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000]
        self.names = names
        self.datafiles = datafiles

    ##runs multiple crossvalidation to determine a set of features for all binary classifiers
    ##returns a union of the selected featrues for later use.
    def run(self):
        train_ratio = 0.8 #what proportion of data you want to use as a test set.

        #here are the datafiles to load from
        #loading data
        test_d_list = []
        train_d_list = []

        #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
        for i in range(len(self.datafiles)):
            #change this load function depending on the type of file you are reading from
            #data, dmy, dmy = load_data(self.datafiles[i], i, True)
            data = genfromtxt(self.datafiles[i], delimiter=",")
            train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
            
        for i in range(len(self.datafiles)):
            for j in range(i+1, len(self.datafiles)):
                Y = hstack((-1*ones(train_d_list[i].shape[0]),ones(train_d_list[j].shape[0])))
                X = vstack((train_d_list[i][:,1:],train_d_list[j][:,1:]))
                self.run_one_vo_cv(Y,X, self.names[i], self.names[j], fold=5)

    def run_one_vo_cv(self, Y, X, label_0, label_1, fold=None):
        if fold == None: #conducting loocv.
            print "LOOCV not yet implemented"
            print "fold needs to be a interger bigger than 0"
            return None
        else:
            print "Doing " + label_0 + "vs" + label_1
            #separating data into k blocks
            Y_blocks = []
            X_blocks = []
            size = Y.shape[0]
            for i in range(fold):
                Y_sub = Y[size*i/fold:size*(i+1)/fold]
                X_sub = X[size*i/fold:size*(i+1)/fold,:]
                Y_blocks.append(Y_sub)
                X_blocks.append(X_sub)

            log = []

            order = self.rank(Y,X)

            #at least use f_min of the features
            for num_feature in self.test:
                error = 0

                ##creating test and validation sets
                for i in range(fold):
                    #picking out validation set.
                    test_Y = Y_blocks[i]
                    test_X = X_blocks[i]
                    train_Y = None
                    train_X = None

                    #assembling other folds into training set.
                    for j in range(fold):
                        if j != i:
                            if train_Y == None:
                                train_Y = Y_blocks[j]
                                train_X = X_blocks[j]
                            else:
                                train_Y = hstack((train_Y, Y_blocks[j]))
                                train_X = vstack((train_X, X_blocks[j]))

                    #start training svm using selected features
                    svm = SVM(train_Y, train_X[:, sorted(order[:num_feature])], run=50, eta=0.01)
                    svm._sgd()
                    error += svm.calculate_errors(test_Y, test_X[:, sorted(order[:num_feature])])

                error = error/fold
                print "error :", error, " #features: ", num_feature

                log.append([error, sorted(order[:num_feature])])
                
        #save result
        if save:
            output = ""
            for entry in log:
                output += str(entry[0]) +"\n"
                output += str(entry[1])[1:-1] +"\n"
            f1 = open("VO_output_18subtypes/"+ label_0+"_vs_"+label_1+".txt", "w")
            f1.write(output)
            f1.close()
        return log

    def rank(self,Y,X):
        #store the cosine value corresponding to a feature in that index
        values = []

        for i in range(X.shape[1]):
            val = abs(sp.spatial.distance.cosine(Y,X[:,i])) 
            values.append((i,val))

        values.sort(key=lambda x: x[1])
        order = []
        for i in range(1,len(values)+1):
            order.append(values[-1*i][0])

        return order

    def get_union_features(self):
        folder = "VO_output/"
        num_features = 17726
        union_features = [0]*num_features
        for i in range(len(self.names)):
            for j in range(i+1, len(self.names)):
                dmy, features = load_rfe(folder+self.names[i]+"_vs_"+self.names[j]+".txt")
                feature = features[10] #this number is arbitrarily set
                print len(feature)
                for t in feature:
                    union_features[t] = union_features[t] + 1
                    
        f1 = open("VO_output/VO_union_features.txt", "w")
        f1.write(str(union_features)[1:-1])
        f1.close()

        count = [0,0,0,0,0,0,0]
        for i in union_features:
            count[int(i)]+=1
        print count

    
'''
Series of function used for feature selection using LASSO
'''
#conducts 5-fold cross validation to find the optimal c.
#preferably implement my own l1 logistic regression.
#returns an array where nonzero features are represented by 0
def select_features_LASSO(Y, X, plot = False, pname = None, save= False, sname=None):
    #conducting cross validation to find optimal c.
    result = cross_validate_LASSO(Y,X,5)
    
    c = []
    error = []
    for i in range(len(result)):
        c.append(log(result[i][0]))
        error.append(result[i][1])

    #plot the result if necessary
    if save:
        f1 = open(sname, "w")
        f1.write(str(c)[1:-1]+"\n"+str(error)[1:-1])
        f1.close()
    if plot:
        fig = plt.figure()
        plt.plot(c,error)
        plt.ylabel("Error")
        plt.xlabel("ln(1/lambda)")
        plt.axis([c[0], c[-1]*1.1, min(error)*0.9, max(error)*1.1])
        plt.savefig(pname)

    #find c that was associated with minimum c
    mindex = 0
    minerror = float("inf")
    for i in range(len(result)):
        if result[i][1] < minerror:
            mindex = i
            minerror = result[i][1]
            
    #run l1 logistic regression again with min_c
    lr = linear_model.LogisticRegression(penalty="l1", C=result[mindex][0])
    lr.fit(X,Y)
    
    print "# of features: ", count_nonzero(lr.coef_)
    print "Optimal c", result[mindex][0]

    ret = ones(lr.coef_.shape[1])
    for i in range(lr.coef_.shape[1]):
        if lr.coef_[0,i] != 0:
            ret[i] = 0

    return ret
    
#using scikit now for l1_logistic regression.
#implement my own l1 logistic later. (subgradient?)
def cross_validate_LASSO(Y, X , fold=None, _c_cur=0.1):
    if fold == None: #conducting loocv.
        print "LOOCV not yet implemented"
        return None
    else:
        _c_inc_ratio = 1.25 # for each iteration rambda is multiply by this number.
        _c_max = 2500000
        
        #separating data into k blocks
        Y_blocks = []
        X_blocks = []
        size = Y.shape[0]
        for i in range(fold):
            Y_sub = Y[size*i/fold:size*(i+1)/fold]
            X_sub = X[size*i/fold:size*(i+1)/fold,:]
            Y_blocks.append(Y_sub)
            X_blocks.append(X_sub)

        #do this until current lambda hits lambda min, which is arbitrarily set.

        c_error_log = []
        
        while _c_cur<_c_max:
            print "Current_c :", _c_cur
            error_sum = 0
            
            for i in range(fold):
                #picking out validation set.
                test_Y = Y_blocks[i]
                test_X = X_blocks[i]
                train_Y = None
                train_X = None

                #assembling other folds into training set.
                for j in range(fold):
                    if j != i:
                        if train_Y == None:
                            train_Y = Y_blocks[j]
                            train_X = X_blocks[j]
                        else:
                            train_Y = hstack((train_Y, Y_blocks[j]))
                            train_X = vstack((train_X, X_blocks[j]))

                #start training
                #write original l1 regression here.
                lr = linear_model.LogisticRegression(penalty="l1", C=_c_cur)
                lr.fit(train_X,train_Y)
                linear_pred = test_X.dot(lr.coef_.T)+lr.intercept_
                predicted_Y = 1-1/(1+exp(linear_pred))

                #calculating a sum of error
                for i in range(test_Y.shape[0]):
                    error_sum += abs(test_Y[i]-predicted_Y[i])[0]

            error_sum = error_sum/Y.shape[0]
            print "Error: ", error_sum
            
            #record error and corresponding lambda
            c_error_log.append((_c_cur, error_sum))
            #update lambda.
            _c_cur = _c_cur * _c_inc_ratio

    #return lambda with min logloss.
    return c_error_log

#conducts feature selection on all binary classfiers, result is saved in a log file.
def extract_features_LASSO():
    train_ratio = 0.8 #what proportion of data you want to use as a test set.

    #here are the datafiles to load from
    names = ["aml", "cALL", "CLL", "mds"]
    datafiles = ["MILES_data/amlMiles2012.csv","MILES_data/cALLMiles2014.csv","MILES_data/CLLMiles2014.csv","MILES_data/mdsMiles2014.csv"]

    #loading data
    test_d_list = []
    train_d_list = []

    #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
    for i in range(len(datafiles)):
        data, dmy, feature_names = load_data(datafiles[i], i, True)
        train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
        test_d_list.append(data[int(train_ratio*data.shape[0]):,:])
    for i in range(len(datafiles)):
        for j in range(i+1, len(datafiles)):
            Y = hstack((zeros(train_d_list[i].shape[0]),ones(train_d_list[j].shape[0])))
            X = vstack((train_d_list[i][:,1:],train_d_list[j][:,1:]))
            select_features_LASSO(Y,X, True, names[i]+"_vs_"+names[j]+"5fold.png", True, names[i]+"_vs_"+names[j]+"5fold_log.txt")

##
def extract_features_LASSO_from_saved_data():
    #loading from log files
    x = loadlog("aml_vs_cALL5fold_log.txt", 0)
    amlvscall = loadlog("aml_vs_cALL5fold_log.txt", 1)
    amlvscll = loadlog("aml_vs_CLL5fold_log.txt", 1)
    amlvsmds = loadlog("aml_vs_mds5fold_log.txt", 1)
    callvscll = loadlog("cALL_vs_CLL5fold_log.txt", 1)
    callvsmds = loadlog("cALL_vs_mds5fold_log.txt", 1)
    cllvsmds = loadlog("CLL_vs_mds5fold_log.txt", 1)

    #finding minimum c for each ones
    c = []
    c.append(exp(find_min(x,amlvscall)))
    c.append(exp(find_min(x,amlvscll)))
    c.append(exp(find_min(x,amlvsmds)))
    c.append(exp(find_min(x,callvscll)))
    c.append(exp(find_min(x,callvsmds)))
    c.append(exp(find_min(x,cllvsmds)))

    #loading data
    train_ratio = 0.8 #what proportion of data you want to use as a test set.

    #here are the datafiles to load from
    names = ["aml", "cALL", "CLL", "mds"]
    datafiles = ["MILES_data/amlMiles2012.csv","MILES_data/cALLMiles2014.csv","MILES_data/CLLMiles2014.csv","MILES_data/mdsMiles2014.csv"]

    #loading data
    test_d_list = []
    train_d_list = []

    #THIS MIGHT NOT BE THE BEST WAY TO SEPARATE TRAINING AND TEST. REVISIT THIS LATER
    for i in range(len(datafiles)):
        data, dmy, feature_names = load_data(datafiles[i], i, True)
        train_d_list.append(data[:int(train_ratio*data.shape[0]),:]) #training data
        test_d_list.append(data[int(train_ratio*data.shape[0]):,:]) #test data

    selected_features = []
    counter = 0
    #zero elements are marked as zero, other elemetns are marked with the number of time selected.
    union_features = zeros(test_d_list[0].shape[1]-1)
    for i in range(len(datafiles)):
        for j in range(i+1, len(datafiles)):
            Y = hstack((zeros(int(train_d_list[i].shape[0]*0.8)),ones(int(train_d_list[j].shape[0]*0.8))))
            X = vstack((train_d_list[i][:int(train_d_list[i].shape[0]*0.8),1:],train_d_list[j][:(train_d_list[j].shape[0]*0.8),1:]))   
            lr = linear_model.LogisticRegression(penalty="l1", C=c[counter])
            lr.fit(X,Y)
            counter += 1
            feature = zeros(lr.coef_.shape[1])
            for t in range(lr.coef_.shape[1]):
                if lr.coef_[0,t] != 0:
                    feature[t] = 1
            union_features = union_features+feature

    result = union_features.tolist()
    f1 = open("LASSO_union_features.txt", "w")
    f1.write(str(result)[1:-1])
    f1.close()

    count = [0,0,0,0,0,0,0]
    print len(result)
    for i in result:
        count[int(i)]+=1
    print count
