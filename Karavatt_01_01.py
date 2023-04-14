# Karavatt, Nikhil Das
# 1002_085_391
# 2023_02_27
# Assignment_01_01

import numpy as np

#Sigmoid Function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#Root Mean Square Error Function
def MSE(Y_pred, Y_true):
  return np.mean((Y_pred - Y_true)**2)

#Function for Adding Bias layer to different input matrices
def add_bias_layer(X):
  return np.concatenate((np.ones((1, X.shape[1])), X))

#initializing weights
def initial_weights(X_train,layers,seed):
  weights=[]
  for i in range (len(layers)):
    np.random.seed(seed)
    if i == 0:
      w= np.random.randn(layers[i],X_train.shape[0]+1) #first layer weights with bias
      weights.append(w)
    else:
      w=np.random.randn(layers[i],layers[i-1]+1) #remaining layers weights with bias
      weights.append(w)
  return weights

#Copying weights function as np.copy creates errors for matrices
def copy_weights(weights):
  copy = []
  for i in weights:
    x = np.copy(i)
    copy.append(x)
  return copy

#Function to Find MSE after nodes getting Activated by sigmoid
def MSE_Activation(weights, X_train_s, Y_train_s):
  input_fed = X_train_s
  activation = None
  for each_layer in weights:
    if activation is not None:
      input_fed = add_bias_layer(input_fed) #adding bias to activation to activation as an input for upcoing layers nodes
    activation = sigmoid(np.dot(each_layer, input_fed)) #using sigmoid as an activation function
    input_fed = activation
  return MSE(activation, Y_train_s)

#Funtion to Update the weights
def updated_weight(weights,X_train_s,Y_train_s,alpha,h):
  Temp_w = copy_weights(weights)
  for i in range(len(weights)):
    for j in range(weights[i].shape[0]):
      for k in range(weights[i][j].shape[0]): #loops to access each weights
        old_val = weights[i][j][k] 
        weights[i][j][k] = old_val + h #adding step size
        mse_plus = MSE_Activation(weights, X_train_s, Y_train_s) #MSE for each weight 
        weights[i][j][k] = old_val - h #subtracting step size
        mse_minus = MSE_Activation(weights, X_train_s, Y_train_s) #MSE for each weight  
        weights[i][j][k] = old_val #making weights back to original value
        dmse_W = (mse_plus - mse_minus)/(2*h) #partial derivative of the weight
        Wnew = weights[i][j][k] - alpha * dmse_W #updated weight
        Temp_w[i][j][k] = Wnew #storing weight in temporary 
  return Temp_w

#To find the output of the Multilayer Neural Network
def output_function(weights,X_test_n,layers):  
  Activated_nodes = [] #storing activated values of the nodes
  for i in range(len(layers)):
    if len(Activated_nodes) == 0: #if at first layer no need to add bias layer
      activated_value = np.dot(weights[i], X_test_n) #finding activated value
      Activated_nodes.append(sigmoid(activated_value)) #find activated value of the nodes for first layer
    else:
      activated_value = np.dot(weights[i], add_bias_layer(Activated_nodes[-1])) #adding bias to the layer as previous layers dont have bias values as inputs 
      Activated_nodes.append(sigmoid(activated_value)) #finding activated values of the nodes for the remaining layers
  return Activated_nodes[-1]

#Function to get list of MSE for each Epoch
def MSES_Epoch(weights, X_test, X_test_n, Y_test):
  MSE_List = []
  for i in range(X_test.shape[1]):
    X_test_s = X_test_n[:,i:i+1] #taking one x test sample at a time
    Y_test_s = Y_test[:,i:i+1] #taking one y test sample at a time
    MSE_List.append(MSE_Activation(weights, X_test_s, Y_test_s)) #making the list of MSE for one epoch
  return MSE_List

#Multi Layer Neural Network Function
def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
  weights = initial_weights(X_train,layers,seed) #added the initialized weights
  output = []
  MSE_Per_Epoch = [] 
  X_train_n = add_bias_layer(X_train) #adding bias to the input values for training
  X_test_n = add_bias_layer(X_test) #adding bias to the input values for testing
  for i in range(epochs):
    for i in range(X_train.shape[1]): 
      X_sample = X_train_n[:,i:i+1] #taking one x train sample at a time
      Y_sample = Y_train[:,i:i+1]   #taking one y train sample at a time
      w = updated_weight(weights,X_sample,Y_sample,alpha,h) 
      weights = copy_weights(w) #updating the weight matrices with the new weights
    MSE_Per_Epoch.append(np.mean(MSES_Epoch(weights, X_test, X_test_n, Y_test))) #list of average MSE for each epoch using test samples
  output = output_function(weights,X_test_n,layers) #output of the multilayer neural network
  return [weights, MSE_Per_Epoch, output] 