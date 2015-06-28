from matplotlib import pyplot as plt
import numpy as np
import math

'''plt.figure()
time = np.array([99.06,207.614,306.7,471.34,624.44,1414.5,2714.24,3846.68,7770.52])
nodes = np.array([4,8,12,16,20,50,100,200,300])
plt.plot(nodes,time)
plt.legend()
plt.xlabel('No of hidden nodes')
plt.ylabel('Training time')
plt.title('Hidden Nodes vs Training Time ( lambdaval = 0.4 ) ')
plt.show() '''

'''plt.figure()
trainAccuracy= np.array([80.27,87.926,91.822,93.25,93.364,93.26,95.54,91.582,93.4])
validationAccuracy = np.array([80.21,87.61,91.51,92.28,92.68,92.45,94.8,91.18,92.76])
testAccuracy = np.array([80.4,87.81,91.83,93.17,92.98,92.73,95.27,91.87,93.17])
nodes = np.array([4,8,12,16,20,50,100,200,300])
plt.plot(nodes,trainAccuracy,color="red",label="red(training set)")
plt.plot(nodes,validationAccuracy,color="blue",label="blue(validation set)")
plt.plot(nodes,testAccuracy,color="green",label="green(test set)")
plt.legend()
plt.xlabel('No of hidden nodes')
plt.ylabel('Accuracy in percentage')
plt.title('Hidden Nodes vs Accuracy ( lambdaval = 0.4 ) ')
plt.show()
'''

'''
plt.figure()
trainAccuracy= np.array([64.784,66.396,70.644,72.428,80.27,62.426,78.23,75.29,60.25,66.41,53.04])
validationAccuracy = np.array([64.12,66.43,70.76,72.13,80.21,62.89,78.21,74.42,59.47,65.68,53.03])
testAccuracy = np.array([64.01,66.43,71.11,72.85,80.4,62.15,78.82,75.58,59.58,65.67,52.82])
lambdaval = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.plot(lambdaval,trainAccuracy,color="red",label="red(training set)")
plt.plot(lambdaval,validationAccuracy,color="blue",label="blue(validation set)")
plt.plot(lambdaval,testAccuracy,color="green",label="green(test set)")
plt.legend()
plt.xlabel('Lambda Values')
plt.ylabel('Accuracy in percentage')
plt.title('Regularization Term (lambda) vs Accuracy ( No of hidden nodes = 4 ) ')
plt.show()
'''
'''
plt.figure()
trainError= np.array([35.216,33.604,29.356,27.572,19.73,37.574,21.77,24.71,39.75,33.59,46.96])
testError = np.array([35.99,33.57,28.89,27.15,19.6,37.85,21.18,24.42,40.42,34.33,47.18])
lambdaval = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.plot(lambdaval,trainError,color="red",label="red(training set)")
plt.plot(lambdaval,testError,color="green",label="green(test set)")
lambdaval.fill(0.4)
trainError[0] = 0
plt.plot(lambdaval,trainError,"b--")
plt.legend()
plt.xlabel('Lambda Values')
plt.ylabel('Error in Percentage')
plt.title('Selecting Lambda - Lambda Values vs Error')
plt.show()
'''

'''dataM= np.array([4.72,4.15,4.46,4.23,5.28,4.07,4.67,4.87,4.54,3.75])
dataB= np.array([5.76,5.12,5.64,5.59,5.54,5.09,5.46,5.87,5.71,5.56])'''

#print np.std(dataM)
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigmoid_z = 1.0/(1.0 + np.exp(-z)) 
    
    return sigmoid_z  #your code here

w = np.array([1.05,-0.52,0.85]);
x = np.array([1,0.44,0.01]);

print sigmoid(np.dot(np.transpose(w),x))


w1 = np.array([0.3,0.5]);

w2 = np.array([5.75 , 0.04]);

w3 = np.array([3.20,0.20]);

w4 = np.array([8.75, -0.5 ,0.02]);


x = np.array([[1,11],[1,18],[1,17],[1,15],[1,9],[1,5],[1,12],[1,19],[1,22],[1,25]]);

xdim = np.array([[1,11,121],[1,18,324],[1,17,289],[1,15,225],[1,9,81],[1,5,25],[1,12,144],[1,19,361],[1,22,484],[1,25,625]]);

y = np.array([6,8,10,4,9,6,3,5,2,10]);




def RMSE(x,y,w):

    rmse = 0.0;

    for inp in range(0,10):
        
            #rmse += math.pow((y[inp] - np.dot(np.transpose(w),x[inp])),2);
            rmse += math.pow((y[inp] - np.dot(np.transpose(w),x[inp])),2) ;
    rmse += np.sum(w**2) ;
    rmse = rmse / 2.0;

    return rmse;

print "likelihood for w1 : " + str(RMSE(x,y,w1));

print "likelihood for w2 : " + str(RMSE(x,y,w2));

print "likelihood for w3 : " + str(RMSE(x,y,w3));

print "likelihood for w4 : " + str(RMSE(xdim,y,w4));


