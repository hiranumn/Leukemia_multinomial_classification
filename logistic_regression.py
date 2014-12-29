'''
CSE 546 Machine Learning
Final Project: Multiclass regression for the subtypes of leukemia

Professor Carlos Guestrin
Project advisor: Harley Montgomery

This module conducts logistic regression. Some parts of the code is imported from assignment 2.
'''

from numpy import *
import math

#Conducts logistic regression on given data X and Y.
class logistic_regressor:
    def __init__(self, _lambda, _eta, label_0, label_1, step_decrease = 1):
        #starting parameters
        self.Y = None #ndarray with true values
        self.X = None #ndarray with feature data
        self.num_samples = 0
        self.num_features = 0
        self._lambda = _lambda #regularization constant for l2 norm
        self._eta = _eta #step_size
        self.step_decrease = step_decrease #step_size is multiplied by this number every step so that it exponentially decades. defaulted to 1.
        self.w = None #starting w if there is one, otherwise it is an ndarray of 0.
        self.w_0 = 0 #defaulted to 0 but it can be something else.
        self.label_0 = label_0
        self.label_1 = label_1

        #termination conditions
        self.th = -1 #stop if logloss is improved less than this. default is -1 which means that the algorith does not stop on this.
        self.lim = 1000  #stop after _lim_ iterations. defaulted to 1000.

        #storing necessary data
        self.logloss_record = []

        #some flags
        self.ready = False
        

    #loads data into the regressor class.
    def load(self, Y, X, w=None, w_0=0):
        self.Y = Y
        self.X = X
        if w == None:
            self.w = zeros(self.X.shape[1])
        else:
            self.w = w
        self.w_0 = w_0
        self.ready = True
        self.num_samples = Y.shape[0]
        self.num_features = X.shape[1]
        

    #changes default termination condition. See above for what th and lim mean.
    def change_termination_condition(self, threshold=None, limit=None):
        if threshold != None:
            self.th = threshold
        if limit != None:
            self.lim = limit
        
    #conducts batch gradient descent.
    def bgd_l2(self, verbose=False):
        #check if the data is loaded before running
        if not self.ready:
            print "Load data before start"
            return None, None, None
        
        #setting parameters before starting iterative steps
        itr = 0
        prev_loss = self.calculate_log_loss()
        self.logloss_record.append(prev_loss)

        while(itr < self.lim): #limiting by iteration count
            if itr%10 == 0 and verbose:
                #printing iteration count every 10 iterations
                print "Iteration: ", itr, " Current log-loss: ", prev_loss 

            #taking a copy of old parameters 
            w_old = self.w.copy()
            w_0_old = self.w_0

            #update w0
            prob = self.calculate_probability()
            self.w_0 = w_0_old+self._eta*((self.Y-prob).sum()/self.num_samples)
            
            #update wk
            self.w = w_old+self._eta*(((-1)*self._lambda*w_old) + ((self.X.transpose()).dot(self.Y-prob))/self.num_samples)

            #keep track of log-loss values
            cur_loss = self.calculate_log_loss()
            self.logloss_record.append(cur_loss)    

            #check to see if we want to continue
            #stop if log loss did not improve more that th.
            if(abs(prev_loss-cur_loss) < self.th):
                return self.w, self.w_0, self.logloss_record

            #update information for the next iteration.
            itr += 1
            prev_loss = cur_loss
        
        return self.w, self.w_0, self.logloss_record

#conducts batch gradient descent. No regularization
    def bgd(self, verbose=False):
        #check if the data is loaded before running
        if not self.ready:
            print "Load data before start"
            return None, None, None
        
        #setting parameters before starting iterative steps
        itr = 0
        prev_loss = self.calculate_log_loss()
        self.logloss_record.append(prev_loss)

        while(itr < self.lim): #limiting by iteration count
            if itr%10 == 0 and verbose:
                #printing iteration count every 10 iterations
                print "Iteration: ", itr, " Current log-loss: ", prev_loss 

            #taking a copy of old parameters 
            w_old = self.w.copy()
            w_0_old = self.w_0

            #update w0
            prob = self.calculate_probability()
            self.w_0 = w_0_old+self._eta*((self.Y-prob).sum()/self.num_samples)
            
            #update wk
            self.w = w_old+self._eta*(((self.X.transpose()).dot(self.Y-prob))/self.num_samples)

            #keep track of log-loss values
            cur_loss = self.calculate_log_loss()
            self.logloss_record.append(cur_loss)    

            #check to see if we want to continue
            #stop if log loss did not improve more that th.
            if(abs(prev_loss-cur_loss) < self.th):
                return self.w, self.w_0, self.logloss_record

            #update information for the next iteration.
            itr += 1
            prev_loss = cur_loss
        
        return self.w, self.w_0, self.logloss_record

#conducts coordinate descent with LASSO regularization.
#this does not work yet
    def subgd_l1(self, verbose=True):
        #check if the data is loaded before running
        if not self.ready:
            print "Load data before start"
            return None, None, None
        
        #setting parameters before starting iterative steps
        itr = 0
        prev_loss = self.calculate_log_loss()
        self.logloss_record.append(prev_loss)

        while(itr < self.lim): #limiting by iteration count
            if itr%10 == 0 and verbose:
                #printing iteration count every 10 iterations
                print "Iteration: ", itr, " Current log-loss: ", prev_loss 

            #taking a copy of old parameters 
            w_old = self.w.copy()
            w_0_old = self.w_0

            #update w0
            prob = self.calculate_probability()
            self.w_0 = w_0_old+self._eta*((self.Y-prob).sum()/self.num_samples)
            
            #update wk
            self.w = w_old+self._eta*((self._lambda*sign(w_old)) + ((self.X.transpose()).dot(self.Y-prob))/self.num_samples)

            #keep track of log-loss values
            cur_loss = self.calculate_log_loss()
            self.logloss_record.append(cur_loss)    

            #check to see if we want to continue
            #stop if log loss did not improve more that th.
            if(abs(prev_loss-cur_loss) < self.th):
                return self.w, self.w_0, self.logloss_record

            #update information for the next iteration.
            itr += 1
            prev_loss = cur_loss
            print self.report_nonzero()
        
        return self.w, self.w_0, self.logloss_record

    #conducts stochastic gradient descent.
    def sgd_l2(self):
        #check if the data is loaded before running
        if not self.ready:
            print "Load data before start"
            return None, None, None
        
        #setting parameters before starting iterative steps
        itr = 0
        prev_loss = self.calculate_log_loss()
        self.logloss_record.append(prev_loss)

        while(itr < self.num_samples*self.lim): #limiting by iteration count
            if itr%10 == 0:
                #printing iteration count every 10 iterations
                print "Iteration: ", itr, " Current log-loss: ", prev_loss 

            #taking a copy of old parameters 
            w_old = self.w.copy()
            w_0_old = self.w_0

            #update w0
            #making a linear predction using the current model
            linear_prediction = (self.X[itr%self.num_samples,:].dot(self.w) + self.w_0)
            prob = 1-1.0/(1.0+exp(linear_prediction))
            self.w_0 = w_0_old+self._eta*(self.Y[itr%self.num_samples]-prob)
            
            #update wk
            exmp = self.X[itr%self.num_samples,:]
            self.w = w_old+self._eta*(((-1)*self._lambda*w_old) + exmp*(self.Y[itr%self.num_samples]-prob))

            #keep track of log-loss values
            cur_loss = self.calculate_log_loss()
            self.logloss_record.append(cur_loss)    

            #check to see if we want to continue
            #stop if log loss did not improve more that th.
            if(abs(prev_loss-cur_loss) < self.th):
                return self.w, self.w_0, self.logloss_record

            #update information for the next iteration.
            itr += 1
            prev_loss = cur_loss
        
        return self.w, self.w_0, self.logloss_record

    #Calculates probability of a sample being 1 in a vector
    def calculate_probability(self):
        #making linear prediction
        linear_prediction = (self.X).dot((self.w)) + self.w_0
        probability = 1.0-1.0/(1.0+exp(linear_prediction))
        return probability

    #Makes prediction that y is 1 with given X with the w it currently has.
    def make_predictions(self, X):
        #making linear prediction
        linear_prediction = (X).dot((self.w)) + self.w_0
        probability = 1.0-1.0/(1.0+exp(linear_prediction))
        return probability
        

    #Calculates logloss of the current model
    def calculate_log_loss(self):
        prob = self.calculate_probability()
        for i in range(self.num_samples):
            if self.Y[i] == 0:
                prob[i] = 1-prob[i]
        prob = log(prob)
        ret = 1/2*(self._lambda)*((self.w).T.dot(self.w)) - (prob.sum()*1.0)/self.num_samples
        return ret

    #Calculates some of squared error (which is just the number of wrong classifications)
    def calcSSE(self):
        #probability of i being 1
        prob = self.calculate_probability()
        num = 0
        #count the number of wrongly classfied samples
        for i in range(self.num_samples):
            #When the target output is 1, if probablity is less than 0.5, it is wrongly classified
            if self.Y[i] == 1:
                if prob[i] < 0.5:
                    num += 1
            #When the target output is 0, if probablity is more than 0.5, it is wrongly classified
            elif self.Y[i] == 0:
                if prob[i] > 0.5:
                    num += 1
        return num

    def report_nonzero(self):
        count = 0
        for i in self.w:
            if self.w[i] == 0:
                count += 1
        return count*1.0
            
        

