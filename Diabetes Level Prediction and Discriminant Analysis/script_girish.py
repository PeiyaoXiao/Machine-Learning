import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as linalg
import math

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    # finding the number of unique classes and list of unique classes
    k = np.unique(y).shape[0]
    listOfClasses = np.unique(y)
    
    # adding the true labels to the matrix
    X = np.concatenate((X,y), axis=1)
    
    # separating the matrices into classes
    # class ONE
    class1 = np.where(X[:,2] == 1.);
    class1Matrix = np.empty([0,2], dtype = float);
    for rowNum in class1:
        class1Matrix = np.vstack((class1Matrix,X[rowNum, 0:2]));
        
    # class TWO
    class2 = np.where(X[:,2] == 2.);
    class2Matrix = np.empty([0,2], dtype = float);
    for rowNum in class2:
        class2Matrix = np.vstack((class2Matrix,X[rowNum, 0:2]));
        
    # class THREE
    class3 = np.where(X[:,2] == 3.);
    class3Matrix = np.empty([0,2], dtype = float);
    for rowNum in class3:
        class3Matrix = np.vstack((class3Matrix,X[rowNum, 0:2]));
        
    # class FOUR
    class4 = np.where(X[:,2] == 4.);
    class4Matrix = np.empty([0,2], dtype = float);
    for rowNum in class4:
        class4Matrix = np.vstack((class4Matrix,X[rowNum, 0:2]));
        
    # class FIVE
    class5 = np.where(X[:,2] == 5.);
    class5Matrix = np.empty([0,2], dtype = float);
    for rowNum in class5:
        class5Matrix = np.vstack((class5Matrix,X[rowNum, 0:2]));
    
    # find the mean of each class
    meanClass1 = class1Matrix.mean(0);
    meanClass2 = class2Matrix.mean(0);
    meanClass3= class3Matrix.mean(0);
    meanClass4 = class4Matrix.mean(0);
    meanClass5 = class5Matrix.mean(0);

    # create the (d X k) mean matrix using vstack
    means = np.vstack((meanClass1, meanClass2, meanClass3, meanClass4, meanClass5));
    # find transpose and bring it to (d X k) format
    means = np.transpose(means);
              
    # find one common covariance matrix
    covmat = np.cov(X[:,0:2], rowvar = 0);
    
    
    return means,covmat
    

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    k = np.unique(y).shape[0]
    listOfClasses = np.unique(y)
    
    # adding the true labels to the matrix
    X = np.concatenate((X,y), axis=1)
    
    # separating the matrices into classes
    # class ONE
    class1 = np.where(X[:,2] == 1.);
    class1Matrix = np.empty([0,2], dtype = float);
    for rowNum in class1:
        class1Matrix = np.vstack((class1Matrix,X[rowNum, 0:2]));
        
    # class TWO
    class2 = np.where(X[:,2] == 2.);
    class2Matrix = np.empty([0,2], dtype = float);
    for rowNum in class2:
        class2Matrix = np.vstack((class2Matrix,X[rowNum, 0:2]));
        
    # class THREE
    class3 = np.where(X[:,2] == 3.);
    class3Matrix = np.empty([0,2], dtype = float);
    for rowNum in class3:
        class3Matrix = np.vstack((class3Matrix,X[rowNum, 0:2]));
        
    # class FOUR
    class4 = np.where(X[:,2] == 4.);
    class4Matrix = np.empty([0,2], dtype = float);
    for rowNum in class4:
        class4Matrix = np.vstack((class4Matrix,X[rowNum, 0:2]));
        
    # class FIVE
    class5 = np.where(X[:,2] == 5.);
    class5Matrix = np.empty([0,2], dtype = float);
    for rowNum in class5:
        class5Matrix = np.vstack((class5Matrix,X[rowNum, 0:2]));
    
    # find the mean of each class
    meanClass1 = class1Matrix.mean(0);
    meanClass2 = class2Matrix.mean(0);
    meanClass3= class3Matrix.mean(0);
    meanClass4 = class4Matrix.mean(0);
    meanClass5 = class5Matrix.mean(0);
    
    # find the covariance of each class
    covClass1 = np.cov(class1Matrix[:,:], rowvar = 0)
    covClass2 = np.cov(class2Matrix[:,:], rowvar = 0)
    covClass3 = np.cov(class3Matrix[:,:], rowvar = 0)
    covClass4 = np.cov(class4Matrix[:,:], rowvar = 0)        
    covClass5 = np.cov(class5Matrix[:,:], rowvar = 0)
    
    # create the (d X k) mean matrix using vstack
    means = np.vstack((meanClass1, meanClass2, meanClass3, meanClass4, meanClass5));
    # find transpose and bring it to (d X k) format
    means = np.transpose(means);
              
    # find one common covariance matrix
    covmats = [covClass1, covClass2, covClass3, covClass4, covClass5];
    
    
    return means,covmats


def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    pdfMatrix = np.empty(5);
    matches = 0
    entries = Xtest.shape[0]
    
    # iterate through all the entries
    for num in range(0, entries):    
        for j in range(0,5):
            pdfMatrix[j] = calculatePDF(means[:,j],covmat,Xtest[num,:]);
        if (ytest[num] == (np.argmax(pdfMatrix)+1)):
            matches += 1;
        pdfMatrix = np.empty(5);
    
    acc = (matches / float(entries)) * 100;
    return acc


def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    pdfMatrix = np.empty(5);
    matches = 0
    entries = Xtest.shape[0]
    
    # iterate through all the entries
    for num in range(0, entries):    
        for j in range(0,5):
            pdfMatrix[j] = calculatePDF(means[:,j],covmats[j],Xtest[num,:]);
        if (ytest[num] == (np.argmax(pdfMatrix)+1)):
            matches += 1;
        pdfMatrix = np.empty(5);
    
    acc = (matches / float(entries)) * 100;    
    
    return acc

    
# this function will calculate the PDF for the gaussian distrubution given parameters (meanClass, covmat)
def calculatePDF(meanClass, covmat, inputData):
    # finding the exponential term of the PDF
    expTerm = (1.0/2.0)* np.dot(np.dot(np.transpose((inputData - meanClass)),linalg.inv(covmat)),(inputData-meanClass));
    # prior Probability is constant for all classes
    priorProb = 1.0/5.0;
    # final pdf formula
    pdf = priorProb / (2*math.pi * math.sqrt(linalg.det(covmat))) * math.exp(-expTerm);
    
    #return the pdf to the calling function
    return pdf
    
    
'''
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeERegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    return Xd
'''
# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))#,encoding = 'latin1')            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

'''
# Problem 2



X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 1.0, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
lambdas = np.linspace(0, 1.0, num=k)
k = 21
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 50}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmax(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lamda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend('No Regularization','Regularization')

'''
