import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
import random

datadir="/home/ahmii/Desktop/DS"
cata=["Cat","Dog"]

resize_img=50
training_data=[]
X_train=[]
Y_train=[]

def create_training_data():
	for catagory in cata:
		path=os.path.join(datadir,catagory)
		classes=cata.index(catagory)
		for img in os.listdir(path):
			try:
				img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				n_arrayy=cv2.resize(img_array,(resize_img,resize_img))
				training_data.append([n_arrayy,classes])
			except Exception as e:
				pass

create_training_data()
random.shuffle(training_data)

for features, label in training_data:
	X_train.append(features)
	Y_train.append(label)

nipx=50*50*1
X_train=np.asarray(X_train)
Y_train=np.asarray(Y_train)

X_train=X_train.reshape(-1,resize_img,resize_img,1)
Y_train=Y_train.reshape(1,len(Y_train))

X_Flatten=X_train.reshape(X_train.shape[0],nipx).T

print(X_Flatten.shape,Y_train.shape)







def sigmoid(z):
	s=1/(1+np.exp(-z))
	return s

def propagate(w, b, X, Y):
        
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HE=RE ### (≈ 2 lines of code)
    z=np.dot(w.T,X)+b
    A = sigmoid(z)# compute activation
    Pos_y=(Y*np.log(A))
    Neg_y=((1-Y)*np.log(1-A))
    cost = np.sum(Pos_y+Neg_y)/-m # compute cost
    
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dz=A-Y
    dw=np.dot(X,dz.T)/m
    db = np.sum(dz)/m
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
  
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w,b,X,Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        
        dw = grads["dw"]
        db = grads["db"]
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w-learning_rate*dw
        b = b-learning_rate*db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs





def predict(w, b, X):
   
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    z=np.dot(w.T,X)+b
    A = sigmoid(z)# compute activation


    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if (A[0,i]<=0.5):
            Y_prediction[0,i]=0
        elif(A[0,i]>0.5):
            Y_prediction[0,i]=1
        pass
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = np.zeros((X_train.shape[0],1)), 0

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w,b,X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
  

    
    d = {"costs": costs, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(X_Flatten, Y_train, num_iterations = 500, learning_rate = 0.005, print_cost = True)
