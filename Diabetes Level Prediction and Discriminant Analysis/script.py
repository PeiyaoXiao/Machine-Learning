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
    
    #plt.scatter(class1Matrix[:,0], class1Matrix[:,1],color="red");
       
    # class TWO
    class2 = np.where(X[:,2] == 2.);
    class2Matrix = np.empty([0,2], dtype = float);
    for rowNum in class2:
        class2Matrix = np.vstack((class2Matrix,X[rowNum, 0:2]));
    
    #plt.scatter(class2Matrix[:,0], class2Matrix[:,1],color="green");

    # class THREE
    class3 = np.where(X[:,2] == 3.);
    class3Matrix = np.empty([0,2], dtype = float);
    for rowNum in class3:
        class3Matrix = np.vstack((class3Matrix,X[rowNum, 0:2]));
    
    #plt.scatter(class3Matrix[:,0], class3Matrix[:,1],color="blue");
   
    # class FOUR
    class4 = np.where(X[:,2] == 4.);
    class4Matrix = np.empty([0,2], dtype = float);
    for rowNum in class4:
        class4Matrix = np.vstack((class4Matrix,X[rowNum, 0:2]));
        
    #plt.scatter(class4Matrix[:,0], class4Matrix[:,1],color="black");

        
    # class FIVE
    class5 = np.where(X[:,2] == 5.);
    class5Matrix = np.empty([0,2], dtype = float);
    for rowNum in class5:
        class5Matrix = np.vstack((class5Matrix,X[rowNum, 0:2]));
        
    #plt.scatter(class5Matrix[:,0], class5Matrix[:,1],color="orange");

    
    
    
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
    plotDiscriminatingBoundary(X,means,covmat,"lda")
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
    plotDiscriminatingBoundary(X,means,covmats , "qda")
    
    return means,covmats



def ldaClass(means,covmat,X):
    pdfMatrix = np.empty(5);
    
    # iterate through all the entries    
    for j in range(0,5):
        pdfMatrix[j] = calculatePDF(means[:,j],covmat,X);
            
    return np.argmax(pdfMatrix)+1        


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
    
def qdaClass(means,covmats,X):
    pdfMatrix = np.empty(5);
    
    for j in range(0,5):
        pdfMatrix[j] = calculatePDF(means[:,j],covmats[j],X);
            
    return np.argmax(pdfMatrix)+1 

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

def plotDiscriminatingBoundary(X,means,covmat,da):
    #print np.min(X[:,0]) ,np.max(X[:,0])
    #print np.min(X[:,1]) ,np.max(X[:,1])
    #x -> x0 , y -> x1
    x = np.arange(0.9 ,14.2 , 0.1)
    y = np.arange(0.9 ,14.2 , 0.1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.empty([xx.shape[1],yy.shape[0]],dtype=float)
    
    plt.figure()
    #for k in range(0,5):
    for i in range(xx.shape[1]):
        for j in range(yy.shape[0]):
            if da == "lda":
                z[i,j] = ldaClass(means , covmat,np.array([xx[0][i],yy[j][0]]) )
            elif da == "qda":
                z[i,j] = qdaClass(means , covmat,np.array([xx[0][i],yy[j][0]]) )

            #print z[i,j]
            if z[i,j] == 1:
                plt.plot(xx[0][i],yy[j][0],"ro")
            elif z[i,j] == 2:
                plt.plot(xx[0][i],yy[j][0],"m^")
            elif z[i,j] == 3:
                plt.plot(xx[0][i],yy[j][0],"go")
            elif z[i,j] == 4:
                plt.plot(xx[0][i],yy[j][0],"ys")
            elif z[i,j] == 5:    
                plt.plot(xx[0][i],yy[j][0],"bo")
    plt.show()          

    
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

def learnOLERegression(X,y):
# Inputs:
# X = N x d
# y = N x 1
# Output:
# w = d x 1
# IMPLEMENT THIS METHOD

    w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X),y));

    return w


def learnRidgeRegression(X,y,lambd):
# Inputs:
# X = N x d
# y = N x 1
# lambd = ridge parameter (scalar)
# Output:
# w = d x 1
# IMPLEMENT THIS METHOD

    ident = np.identity(X.shape[1]);
    entries = X.shape[0];
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X),X) + (lambd * entries * ident) ), np.dot(np.transpose(X), y));

    return w
    
    
def testOLERegression(w,Xtest,ytest):
# Inputs:
# w = d x 1
# Xtest = N x d
# ytest = X x 1
# Output:
# rmse
# IMPLEMENT THIS METHOD
    rmse = 0;
    entries = Xtest.shape[0];
    for num in range(0,entries):   
        rmse += (ytest[num] - np.dot(np.transpose(w), Xtest[num])) ** 2;
    rmse = float(sqrt(rmse) * 1/entries);
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = w.reshape(w.shape[0],1)
    error = (1.0/(2.0*X.shape[0]))*np.dot(np.transpose(y - np.dot(X,w)),(y - np.dot(X,w)))  + (1.0/2.0 * lambd * (np.dot(np.transpose(w),w)))                                            
    
    #1/2N * (y-xw)T * (y-xw) + 1/2 * lambda * wT * w
    error_grad = (1.0/X.shape[0])*(np.dot(np.transpose(w) ,np.dot(np.transpose(X) , X) )  - np.dot(np.transpose(y),X)) + lambd*np.transpose(w)
    
    return error, error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.empty([x.shape[0],p+1],dtype=float)
    Xd[:,0] = 1
    for inputs in range (x.shape[0]):
        for numattr in range(1,p+1):
            Xd[inputs,numattr] = math.pow(x[inputs],numattr)
    #print Xd
    return Xd

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




# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))#,encoding = 'latin1')   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_train  = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_i_train = testOLERegression(w_i,X_i,y)

print('----Train Data---')
print('RMSE without intercept '+str(mle_train))
print('RMSE with intercept '+str(mle_i_train))

print('----Test Data---')
print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, .004, num=k)
#lambdas = np.arange(0, .01 , .001)

i = 0
rmses3 = np.zeros((lambdas.shape[0],1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
print "Best lambda:" + str(lambdas[np.argmin(rmses3)])
plt.figure()
plt.plot(lambdas,rmses3)
plt.show()

print math.sqrt(np.sum(np.square(w_i)))

print math.sqrt(np.sum(np.square(w_l)))


# Problem 4
k = 101
lambdas = np.linspace(0, .004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    rmses4[i] = testOLERegression(w_l.x,Xtest_i,ytest)
    i = i + 1
print "Optimal Lambda: " +  str(lambdas[np.argmin(rmses4)])
plt.figure()
plt.plot(lambdas,rmses4)
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)    


plt.figure()
plt.plot(range(pmax),rmses5)
plt.legend(['NoRegularization','Regularization'] )
plt.show() 