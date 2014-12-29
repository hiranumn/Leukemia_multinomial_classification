'''
Naozumi Hiranuma
adapted from hw3
Implementing svm!!

'''

from numpy import *
import sys
import matplotlib.pyplot as plt
import math

def loaddata(fname):
    data = loadtxt(fname,skiprows=1, delimiter=",")
    y = data[:,0]
    X = data[:,1:]
    nm = sqrt(sum(X * X, axis=1))
    X = X / nm[:,None]
    return y, X

class SVM:
    def __init__(self, Y, X, w=None, w_0=0, run=50, eta=0.03, c=1, label_0="", label_1=""):
        #data and current weights
        self.Y = Y
        self.X = X
        if w == None:
            self.w = zeros(X.shape[1])
        self.w_0 = 0
        self._eta = eta
        self._c = c

        self.run = run

        self.N = X.shape[0]
        self.D = X.shape[1]

        self.label_0 = label_0 
        self.label_1 = label_1 

        self.loss_record = []

    def _sgd(self):
        count = 0
        #this outer loop is basically one pass
        self.loss_record.append(self.calculate_loss(True))
        while(count < self.run):
            #this is each sample.
            for i in range(self.N):
                #check if the prediction is right
                if self.Y[i]*(self.X[i,:].dot(self.w)+self.w_0) <= 1:
                    mistake = 1
                else:
                    mistake = 0
                
                #updating w_0
                self.w_0 = self.w_0-self._eta*(-1*mistake*self.Y[i])
                #updating w
                temp = (2.0/(self.N*self._c))*self.w #This integer division took me 3 hrs to notice.
                self.w = self.w-self._eta*(temp-mistake*self.Y[i]*self.X[i,:])
                
            loss = self.calculate_loss(True)
            #print "Pass:", count+1, " Loss:", loss
            self.loss_record.append(loss)

            count += 1

    def make_predictions(self, X):
        preds = X.dot(self.w)+self.w_0
        return preds
        

    def plot_loss(self, fname):
        fig = plt.figure()
        x = range(len(self.loss_record)-1)
        plt.plot(x, self.loss_record[1:])
        plt.ylabel("loss")
        plt.xlabel("# pass")
        plt.axis([x[0], x[-1]*1.05, min(self.loss_record[1:])*0.9, max(self.loss_record[1:])*1.1])
        plt.savefig(fname)
        #plt.show()

    def calculate_loss(self, mode):
        ret = (1.0/(self.N*self._c))*(self.w.dot(self.w))

        #We can possibly make this faster
        if not mode:
            temp = 0
            for i in range(self.N):
                temp += max(1-self.Y[i]*(self.X[i,:].dot(self.w)+self.w_0),0)
            ret += temp*1.0/self.N
            return ret
        
        else:
            temp = 1-self.Y*(self.X.dot(self.w)+self.w_0)
            temp = temp.reshape((self.N,1))
            temp2 = zeros(self.N).reshape((self.N,1))
            temp3 = hstack((temp,temp2))
            temp4 = amax(temp3, axis=1)
            ret += temp4.sum()/self.N
            return ret

    def calculate_magnitude(self):
        mag = math.sqrt(self.w.dot(self.w))
        return mag

    #calculating SSE.
    def calculate_errors(self, Y, X):
        preds = X.dot(self.w)+self.w_0
        #(number of samples wrongly classified)/(total number of samples) 
        wrong = 0
        correct = 0
        
        for i in range(Y.shape[0]):
            if preds[i] < 0:
                if Y[i] == -1:
                    correct += 1
                else:
                    wrong += 1
            else:
                if Y[i] == 1:
                    correct += 1
                else:
                    wrong += 1

        ret = wrong*1.0/(wrong+correct)
        return ret
            
def main():
    print "HW3 coding SVMs"
    trainY, trainX = loaddata("validation.csv")
    testY, testX = loaddata("test.csv")

    svm = SVM(trainY, trainX, c=100)
    svm.sgd()
    svm.plot_loss("loss2.png")
    print "C=", svm._c, ", eta=", svm._eta
    print "(a) ", svm.calculate_errors(trainY, trainX)
    print "(b) ", svm.calculate_errors(testY, testX)
    print "(c) ", svm.calculate_magnitude()

if __name__ == "__main__":
    main()
