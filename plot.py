import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.sparse as sparse
import math

def load(filename, line):
    fo = open(filename)
    lines = fo.readlines()
    c = lines[line].split(",")
    for i in range(len(c)):
        c[i] = float(c[i])
    fo.close()
    return c

"This is used to load a file created by rfe (u can use it for vo files too)"
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

def plot_lasso():
    #x-axis log transformed 1/lambda
    folder = "LASSO_output_4subtypes/"
    x = load(folder+"aml_vs_cALL5fold_log.txt", 0)
    amlvscall = load(folder+"aml_vs_cALL5fold_log.txt", 1)
    amlvscll = load(folder+"aml_vs_CLL5fold_log.txt", 1)
    amlvsmds = load(folder+"aml_vs_mds5fold_log.txt", 1)
    callvscll = load(folder+"cALL_vs_CLL5fold_log.txt", 1)
    callvsmds = load(folder+"cALL_vs_mds5fold_log.txt", 1)
    cllvsmds = load(folder+"CLL_vs_mds5fold_log.txt", 1)
    
    
    fig = plt.figure()
    a = plt.plot(x, amlvscall)
    b = plt.plot(x, amlvscll)
    c = plt.plot(x, amlvsmds)
    d = plt.plot(x, callvscll)
    e = plt.plot(x, callvsmds)
    f = plt.plot(x, cllvsmds)
    plt.ylabel("Error")
    plt.xlabel("ln(1/lambda)")
    #plt.axis([x[-1], x[1], min(min(amlvscall),min(amlvscll),min(amlvsmds),min(callvscll),min(callvsmds),min(cllvsmds))*0.9, max(max(amlvscall),max(amlvscll),max(amlvsmds),max(callvscll),max(callvsmds),max(cllvsmds))*1.1])
    plt.axis([x[0], x[-1], 0, 0.3])
    plt.legend([a[0],b[0],c[0],d[0],e[0],f[0]],["aml vs cALL","aml vs CLL","aml vs mds","cALL vs CLL","cALL vs mds", "CLL vs mds"],prop={'size':10})
    plt.savefig(folder+"5-fold_cv_error.png")
    plt.show()

def plot_rfe():
    
    folder = "RFE_output_4subtypes/"
    amlvscall, features = load_rfe(folder+"aml_vs_cALL.txt")
    amlvscll, dmy = load_rfe(folder+"aml_vs_CLL.txt")
    amlvsmds, dmy = load_rfe(folder+"aml_vs_mds.txt")
    callvscll, dmy = load_rfe(folder+"cALL_vs_CLL.txt")
    callvsmds, dmy = load_rfe(folder+"cALL_vs_mds.txt")
    cllvsmds, dmy = load_rfe(folder+"CLL_vs_mds.txt")

    xaxis = []
    for feature in features:
        xaxis.append(math.log(len(feature)))
        #xaxis.append(len(feature))

    fig = plt.figure()
    a = plt.plot(xaxis, amlvscall)
    b = plt.plot(xaxis, amlvscll)
    c = plt.plot(xaxis, amlvsmds)
    d = plt.plot(xaxis, callvscll)
    e = plt.plot(xaxis, callvsmds)
    f = plt.plot(xaxis, cllvsmds)
    
    plt.ylabel("Error")
    plt.xlabel("ln(#features)")
    
    #plt.axis([xaxis[0], xaxis[-1], min(min(amlvscall),min(amlvscll),min(amlvsmds),min(callvscll),min(callvsmds),min(cllvsmds))*0.9, max(max(amlvscall),max(amlvscll),max(amlvsmds),max(callvscll),max(callvsmds),max(cllvsmds))*1.1])
    plt.axis([xaxis[-1], xaxis[0], 0, 0.3])
    plt.legend([a[0],b[0],c[0],d[0],e[0],f[0]],["aml vs cALL","aml vs CLL","aml vs mds","cALL vs CLL","cALL vs mds", "CLL vs mds"],prop={'size':10})
    plt.savefig(folder+"RFE_SSE.png")
    plt.show()

def plot_vo():
    
    folder = "VO_output_4subtypes/"
    amlvscall, features = load_rfe(folder+"aml_vs_cALL.txt")
    amlvscll, dmy = load_rfe(folder+"aml_vs_CLL.txt")
    amlvsmds, dmy = load_rfe(folder+"aml_vs_mds.txt")
    callvscll, dmy = load_rfe(folder+"cALL_vs_CLL.txt")
    callvsmds, dmy = load_rfe(folder+"cALL_vs_mds.txt")
    cllvsmds, dmy = load_rfe(folder+"CLL_vs_mds.txt")

    xaxis = []
    for feature in features:
        xaxis.append(math.log(len(feature)))
        #xaxis.append(len(feature))

    fig = plt.figure()
    a = plt.plot(xaxis, amlvscall)
    b = plt.plot(xaxis, amlvscll)
    c = plt.plot(xaxis, amlvsmds)
    d = plt.plot(xaxis, callvscll)
    e = plt.plot(xaxis, callvsmds)
    f = plt.plot(xaxis, cllvsmds)
    
    plt.ylabel("Error")
    plt.xlabel("ln(#features)")
    
    #plt.axis([xaxis[0], xaxis[-1], min(min(amlvscall),min(amlvscll),min(amlvsmds),min(callvscll),min(callvsmds),min(cllvsmds))*0.9, max(max(amlvscall),max(amlvscll),max(amlvsmds),max(callvscll),max(callvsmds),max(cllvsmds))*1.1])
    plt.axis([xaxis[0], xaxis[-1], 0, 0.3])
    plt.legend([a[0],b[0],c[0],d[0],e[0],f[0]],["aml vs cALL","aml vs CLL","aml vs mds","cALL vs CLL","cALL vs mds", "CLL vs mds"],prop={'size':10})
    plt.savefig(folder+"VO_SSE.png")
    plt.show()
    
    
def main():
     plot_rfe()
    
        

    

if __name__ == "__main__":
    main()
