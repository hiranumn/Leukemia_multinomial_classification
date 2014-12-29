'''
CSE 546 Machine Learning
Final Project: Multiclass regression for the subtypes of leukemia

Professor Carlos Guestrin
Project advisor: Harley Montgomery

This module conducts logistic regression. Some parts of the code is imported from assignment 2.
'''

from numpy import *
import math

#Conducts softmax regression on given data X and Y.
class softmax_regressor:
    def __init__(self, _lambda, _eta, labels, step_decrease = 1):
        #starting parameters
        self.Y = None #ndarray with true values
        self.X = None #ndarray with feature data
        self.num_samples = 0
        self.num_features = 0
        self._lambda = _lambda #regularization constant for l2 norm
        self._eta = _eta #step_size
        self.step_decrease = step_decrease #step_size is multiplied by this number every step so that it exponentially decades. defaulted to 1.
        self.w = None #starting w if there is one, otherwise it is an ndarray of 0.
        self.w_0 = None #defaulted to 0 but it can be something else.
        self.labels = labels

        #termination conditions
        self.th = -1 #stop if logloss is improved less than this. default is -1 which means that the algorith does not stop on this.
        self.lim = 1000  #stop after _lim_ iterations. defaulted to 1000.

        #storing necessary data
        self.logloss_record = []

        #some flags
        self.ready = False
        

    #loads data into the regressor class.
    def load(self, Y, X, w=None, w_0=None):
        self.Y = Y
        self.X = X
        if w == None:
            self.w = zeros(self.X.shape[1]*(len(self.labels)-1)).reshape((self.X.shape[1],len(self.labels)-1))
        else:
            self.w = w
        if w_0 == None:
            self.w_0 = zeros(len(self.labels)-1)
        else:
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
            if itr%1 == 0 and verbose:
                #printing iteration count every 10 iterations
                print "Iteration: ", itr, " Current log-loss: ", prev_loss 

            #taking a copy of old parameters 
            w_old = self.w.copy()
            w_0_old = self.w_0

            #update w0s (there are k of them)
            prob = self.calculate_probability()[:,1:]

            true_matrix = zeros(prob.shape[0]*prob.shape[1]).reshape((prob.shape[0],prob.shape[1]))
            for i in range(self.num_samples):
                if(self.Y[i] != 0):
                    true_matrix[i,self.Y[i]-1] = 1
            #print "true matrix", true_matrix.shape

            #calculating the overall error
            error = true_matrix-prob

            #print "error matrix", error.shape

            #update w_0s by summing across points
            self.w_0 = w_0_old+(self._eta/self.num_samples)*sum(error,axis=0)
            
            #update wk
            self.w = w_old+(self._eta*(((self.X.transpose()).dot(error))/self.num_samples))

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
        #making linear prediction.
        #using first class as a normalizer.
        #print "probability calculation starts"
        linear_prediction = (self.X).dot((self.w)) + self.w_0
        #print linear_prediction.shape
    
        expo = exp(linear_prediction)
        #print expo.shape

        denom = 1+sum(expo, axis=1)
        #print denom.shape
        temp = 1/denom
        
        probability = divide(expo.T,denom.T).T
        #print probability.shape
        #print "probability calculation ends \n"
        probability = hstack((temp.reshape((temp.shape[0],1)),probability))

        return probability

    #Makes prediction that y is 1 with given X with the w it currently has.
    def make_predictions(self, X):
        #making linear prediction
        linear_prediction = (X).dot((self.w)) + self.w_0
        expo = exp(linear_prediction)
        denom = 1+sum(expo, axis=1)
        temp = 1/denom
        probability = divide(expo.T,denom.T).T
        probability = hstack((temp.reshape((temp.shape[0],1)),probability))
        return probability

    #Calculates logloss of the current model
    def calculate_log_loss(self):
        prob = self.calculate_probability()
        prob_t = zeros(prob.shape[0])
        for i in range(self.num_samples):
            prob_t[i] = prob[i,self.Y[i]]
        prob_t = log(prob_t)

        #l2 regularization and teh sum of log prob of true value
        ret = 1/2*(self._lambda)*(sum(multiply(self.w,self.w))) - (prob_t.sum()*1.0)/self.num_samples
        return ret
