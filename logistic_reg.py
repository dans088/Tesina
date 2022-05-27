import numpy as np
import pandas as pd
from PIL import Image
from sklearn import metrics
import random
from scipy.special import expit
from matplotlib import image
from matplotlib import pyplot
from sklearn.model_selection import KFold
import os
from os import listdir


data = image.imread('tileNON_000.png')
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()


class1_path = 'DataCLASS1'
class2_path = 'DataCLASS2'

rows = (len(os.listdir((class1_path))))
cols = data2.shape[0]
print(rows)

def load_dataset(path): 
    loaded_images = []
    for filename in listdir(path):
        # load image
        img_data = image.imread(path + '/' + filename)
        #convert nxmx3 matrix to 1 *(nxm) vector
        img_dataVector = img_data.transpose(2,0,1).reshape(-1)
        # store loaded image
        loaded_images.append(img_dataVector)
        print('> loaded %s %s' % (filename, img_data.shape))
    return loaded_images


class1List = load_dataset(class1_path)
class2List = load_dataset(class2_path)


conSet = class1List + class2List
len(conSet)

#Generate vector with '1' and '0' labels for fertile and unfertile respectively
labelVector = [0 if i<rows else 1 for i in range(rows*2)]

#shuffle dataset to eliminate bias
dataset = list(enumerate(conSet))
random.shuffle(dataset)
indices, conSet = zip(*dataset)

#keep track of labeled indices
for i in range(rows*2):
    labelVector[i] = labelVector[indices[i]] 

#convert labelVector into numpy array
dataset = np.array(conSet)
Y = np.array(labelVector)

# prepare the cross-validation procedure
kf = KFold(n_splits=8)
kf.get_n_splits(dataset)


def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s


#weights and bias vector
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

#hypthesis function that calculates cost 
def h(w, b, X, Y):
    #Find the number of training data
    m = X.shape[1]
    
    #Calculate the predicted output
    #fix the log 0 error with '@' operator
    A = (sigmoid(expit(w.T @ X) + b))
    
    #Calculate cost with cost entropy function 
    cost = -1/m * np.sum(Y*np.log(A) + (1-Y) * np.log(1-A))
    #Calculate the gradients
    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)
        
    grads = {"dw": dw,
            "db": db}
    
    return grads, cost


#gradient descent function
def GD(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    #propagate function will run for a number of iterations    
    for i in range(num_iterations):
        grads, cost = h(w, b, X, Y)    
        dw = grads["dw"]
        db = grads["db"]
        
        #Updating w and b by deducting the dw 
        #and db times learning rate from the previous 
        #w and b
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        '''
        #Record the cost function value for each 5 iterations        
        if i % 5 == 0:
            costs.append(cost)
        '''
        costs.append(cost)
            
    #The final updated parameters     
    params = {"w": w,
            "b": b}
    
    #The final updated gradients    
    grads = {"dw": dw,
            "db": db}
    
    return params, grads, costs


#prediction function
def predict(w, b, X):
    if(X.ndim != 2):
        m = X.shape[0]
    else:  
        m = X.shape[1]
    
    #Initializing an aray of zeros which has a size of the input
    #These zeros will be replaced by the predicted output 
    Y_prediction = np.zeros((1, m))
    
    
    w = w.reshape(X.shape[0], 1)
    
    #Calculating the predicted output  
    #This will return the values from 0 to 1
    A = sigmoid(np.dot(w.T, X) + b)
    
    
    #Iterating through A and predict an 1 if the value of A
    #is greater than 0.5 and zero otherwise
    if(X.ndim != 2):
       Y_prediction = (A > 0.5) * 1
    else:
        for i in range(A.shape[1]):
            Y_prediction[:, i] = (A[:, i] > 0.5) * 1
        
    return Y_prediction

#callable function from main
def model(X, Y, num_iterations, learning_rate, Y_prediction_train_total, Y_prediction_test_total, j, k):
    
    scores = []
    
    for train_index, test_index in kf.split(X,Y):
        
        #Initializing the w and b as zeros
        w, b = initialize_with_zeros(X.shape[1]) 
    

        X_train, X_test = X[train_index].T, X[test_index].T
        Y_train, Y_test = Y[train_index].T, Y[test_index].T
         
        parameters, grads, costs = GD(w, b, X_train, Y_train, num_iterations, learning_rate)
        w = parameters["w"]
        b = parameters["b"]
        
    
        # Predicting the output for both test and training set 
        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)
        
        print(Y_prediction_test)
        
        #fill in complete predictions check Overfit
        for i in range(Y_prediction_test.shape[1]):
            Y_prediction_test_total[j] = Y_prediction_test[:, i]
            j+=1
              
        for i in range(Y_prediction_train.shape[1]):
            Y_prediction_train_total[k] = Y_prediction_train[:, i]
            k += 1
        
        
        #Calculating the training and test set accuracy by comparing
        #the predicted output and the original output
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        scores.append(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)
    
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "Y_prediction_test_total": Y_prediction_test_total, 
             "Y_prediction_train_total" : Y_prediction_train_total, 
             "w" : w, 
             "b" : b,
             "scores": scores,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}
    
    return d


def predictImage(imageName):
    data = image.imread(imageName)
    img_dataVector = data.transpose(2,0,1).reshape(-1).T
    prediction = predict(d['w'], d['b'], img_dataVector)
    print(prediction)

#fill np arrays with zeros for overfitting test
Y_prediction_test_total = np.zeros(48*8)
Y_prediction_train_total= np.zeros(335*8)

#array counters
j = 0
k = 0


#call model function 
d = model(dataset, Y, 100, 0.005, Y_prediction_train_total, Y_prediction_test_total, j, k)

#check k-fold scores
print(d['scores'])

#plot cost function
pyplot.figure(figsize=(7,5))
pyplot.scatter(x = range(len(d['costs'])), y = d['costs'], color='black')
pyplot.title('Scatter Plot of Cost Functions', fontsize=18)
pyplot.ylabel('Costs', fontsize=12)
pyplot.show()

#get model average score
total = sum(d['scores'])
length = len(d['scores'])
average = total/length

print(average)


#plot ROC curve for test set
y_arr = np.array(d['Y_prediction_test_total'])
fpr, tpr, _ = metrics.roc_curve(Y,  y_arr[:382])

#create ROC curve
pyplot.plot(fpr,tpr)
pyplot.ylabel('True Positive Rate')
pyplot.xlabel('False Positive Rate')
pyplot.show()

#plot ROC curve for train set
y_arr2 = np.array(d['Y_prediction_train'][0])
fpr2, tpr2, _ = metrics.roc_curve(Y[:335],  y_arr2)

#create ROC curve
pyplot.plot(fpr2,tpr2)
pyplot.ylabel('True Positive Rate')
pyplot.xlabel('False Positive Rate')
pyplot.show()

#test model with sample image
name = input("Enter image name: ")
tile_path = 'BOTH_CLASS' + '/' + name
img = image.imread(tile_path)
pyplot.imshow(img)
pyplot.show()

result = predictImage(tile_path)
