# -*- coding: utf-8 -*-
"""
################################################################################################
Created on Sun Apr 15 16:45:07 2019
  Answer to Home work2
                    Instruction
                  Select all and run it. You will get the answer to all the questions 
                  by defoult I set it to use the Adam Optimizer, but you can change the
                  optimization_option variable to either 1 or 2 which stands for SGD and
                  Adagrad optimizer.
@author: gaddisa olani
"""

"""
##########################################################################################################
########                        NOTICE:                                                                 ##
########    I IMPLEMENTED ALL THE OPTIMAZATION METHODS, NETWORK DESIGN, ACTIVATION FUNCTIONS, TRAINING ###
########                 IN PURE PYTHON FROM SCRATCH                                                    ##
########                     THANK YOU                                                                  ##
##########################################################################################################

"""





#list of imported libraries for this homework
import numpy as np
import pandas as pd
from tqdm import trange #added to show the progress of learning at each iteration
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#to make it clear for you I use most familiar terms Layer, Dense, Relu

class Layer:
    def __init__(self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        self.weights = np.zeros(shape=(input.shape[1], 10))
        self.bias = np.zeros(shape=(10,))
        pass
    
    def forward_pass(self, input):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, 10]
        """
        output = np.matmul(input, self.weights) + self.bias
        return output

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # initialize weights with small random numbers. I use normal initialization
        self.weights = np.random.randn(input_units, output_units)*0.01
        self.biases = np.zeros(output_units)
        
    def forward_pass(self,input):
        return np.matmul(input, self.weights) + self.biases
      
    def backward(self,input,grad_output):
        # use a chain rule to calculate the weight and bias 
        grad_input = np.dot(grad_output,np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        grad_weights = np.transpose(np.dot(np.transpose(grad_output),input))
        grad_biases = np.sum(grad_output, axis = 0)
        
        """
        Optimzation Using Adam
        call our adam optimizer class
        """
        if optimization_option==0:
            #adam optimizer
            adam_optimizer_weights = AdamOptimizer(self.weights, alpha=0.01)
            self.weights = adam_optimizer_weights.backward_pass(grad_weights)
        
        
            adam_optimizer_biases = AdamOptimizer(self.biases, alpha=0.01)
            self.biases = adam_optimizer_biases.backward_pass(grad_biases)
       
        elif optimization_option==1:
            #SGD optimizer
            self.weights=self.weights-self.learning_rate * grad_weights
            self.biases = self.biases - self.learning_rate * grad_biases
            
        else:
            #which means adagrad implementation
            weight_cache= np.square(grad_weights)
            self.weights=self.weights - 0.01 * grad_weights / (10**-6 + np.sqrt(weight_cache))
        
            bias_cache= np.square(grad_biases)
            self.biases=self.biases - 0.01 * grad_biases / (10**-6 + np.sqrt(bias_cache))
            
            
        return grad_input

class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass
    
    def forward_pass(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0,input)

    def backward(self, input, grad_output):
        """Compute gradientof this activation"""
        relu_grad = input > 0
        return grad_output*relu_grad 

#implement AdamOptimizer algorith, I initialize those parameter 
#according to the best setting defined by most data scientists 
class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.theta = weights
        
    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        
        #self.theta = self.theta - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))

        self.theta = self.theta - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.theta


def crossentropy_loss_calculation(last_layer_act_value,reference_answers):
    """
    Calculate the crossentropy loss for each minibatches
    
    using the simplified loss formula (to avoid division) of minibatch cross entropy loss
    
    """
    a = last_layer_act_value[np.arange(len(last_layer_act_value)),reference_answers]
    
    cross_entropy = - a + np.log(np.sum(np.exp(last_layer_act_value),axis=-1))
    
    return cross_entropy

def gradientof_crossentropy_loss_calculation(last_layer_act_value,reference_answers):
    """Compute crossentropy gradient from last_layer_act_value"""
    
    a = np.zeros_like(last_layer_act_value)
    a[np.arange(len(last_layer_act_value)),reference_answers] = 1
    
    softmax = np.exp(last_layer_act_value) / np.exp(last_layer_act_value).sum(axis=-1,keepdims=True)
    
    return (- a + softmax) / last_layer_act_value.shape[0]


#after the grid search we found the following network structure
def create_the_network(hidden_layers=1,hidden_unit=100):
    #input layer
    network.append(Dense(X_train.shape[1],hidden_unit))
    network.append(ReLU())
    
    #add n hidden layers
    for i in range(hidden_layers):
        network.append(Dense(hidden_unit,hidden_unit))
        network.append(ReLU())
    
    #outputlayer
    network.append(Dense(hidden_unit,6))


def forward_pass(network, X):
    """
    Compute activations of all network layers by applying them sequentially.
    Return a list of activations for each layer. 
    using a python list I store the gradient of all layers in the form
    of table to avoid repetitive computation during backpropagation
    """
    activations = []
    #input = X
    for i in range(len(network)):
        activations.append(network[i].forward_pass(X))
        X = network[i].forward_pass(X)
        
    assert len(activations) == len(network)
    #return the activation of each layer
    return activations


def train(network,X,y):
    """
    Train network on a given batch of X and y.
    Step 1:  run forward_pass to get all layer activations.
    Step 2:  Then you can run layer.backward going from last to first layer.
    After you called backward for all layers, all Dense layers have already made one gradient step.
    """
    
    # Step 1: Get the layer activations
    layer_activations = forward_pass(network,X)
    
    last_layer_act_value = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    loss = crossentropy_loss_calculation(last_layer_act_value,y)
    
    loss_of_the_gradient = gradientof_crossentropy_loss_calculation(last_layer_act_value,y)
    
    """
    The lenght of my network is five: 
        Input layer (dense)
        Activation Layer (Relu)
        Dense Layer()
        Activation Layer (Relu)
        Output Dense Layer(100,6)

    """
    
    for i in range(1, len(network)):
        loss_of_the_gradient = network[len(network) - i].backward(layer_activations[len(network) - i - 1], loss_of_the_gradient)

    return np.mean(loss)

"""
 #######################################################################################################################
 ## tO FIND THE OPTIMAL PARAMETERS GRIDSEARCH IMPLEMENTATION OF sklearn.model_selection WAS USED    ####################
 ##     I RUN IT ONE TIME BECOUSE IT TAKES TO MUCH TIME ON MY PC TO PROVIDE THE RESULT              ####################
 ##     THE FINAL OUTPUT OF RUNNING A GRID SEARCH WAS DISCUSSED IN MY REPORT                        ####################
 #######################################################################################################################

"""

def do_the_GridSeach(model,X,y):
    # define the grid search parameters
    hidden_layers=[1,2,3,4,5]
    hidden_unit=[10,50,100,150,200]
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    batch_size = [32, 64, 128, 256, 512,1024]
    epochs = [10, 50, 100,140,200,500,1000]
    param_grid = dict(hidden_layers=hidden_layers,hidden_unit=hidden_unit,learn_rate=learn_rate,batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
#create minibatch size 
def create_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
#give unseen dataset_ test set perform a  classification based on the learned weights and bias term
def prediction_on_test_dataset(X_test):
    last_layer_act_value = forward_pass(network,X)[-1]
    predicted_values=last_layer_act_value.argmax(axis=-1)
    
    #since we already subtracted one from the class label, know we add one to the final output
    predicted_values+=1
    #save the prediction resut to a text file by a name ....
    pd.DataFrame(predicted_values).to_csv("106062867_Ac.csv")

def precision_recall_f1(X_val,y_val):
    print("#####################################################################")
    print("Answer for question 1) ii")
    predicted_values,x=predict(network,X_val)
    target_names = ['class dws', 'class ups', 'class sit','class std','class wlk','class jog']
    print(classification_report(y_val, predicted_values, digits=4,target_names=target_names))

#prediction a training and validation phase
def predict(network,X):
    """
    Compute network predictions.
    """
    last_layer_act_value = forward_pass(network,X)[-1]
    
    current_loss=last_layer_act_value
    
    #pass two parameters one the classification result and also the loss value
    
    return last_layer_act_value.argmax(axis=-1),current_loss


#everything that we need to train our network is packed her
def begin_the_training():
    number_of_epochs=140
    for epoch in range(number_of_epochs):
        for x_batch,y_batch in create_minibatches(X_train,y_train,batchsize=batch_size,shuffle=True):
            train(network,x_batch,y_batch)
        
        
        """
        Record the training accuracy, validation accuracy
        Record the training loss and validation loss
        Finally plot the curve at the end of the epochs
        """
        training_predicted_value,training_loss=predict(network,X_train)
        train_accuracy_log.append(np.mean(training_predicted_value==y_train))
        train_loss.append(np.mean(crossentropy_loss_calculation(training_loss,y_train)))
        
        val_predicted_value,validation_loss=predict(network,X_val)
        val_accuracy_log.append(np.mean(val_predicted_value==y_val))
        val_loss.append(np.mean(crossentropy_loss_calculation(validation_loss,y_val)))
        
        print("Epoch",epoch)
        print("Train accuracy:",train_accuracy_log[-1])
        print("Val accuracy:",val_accuracy_log[-1])
   
   
    #plot the training  accuracy vs validation accuracy
    plt.plot(train_accuracy_log,label='train accuracy')
    plt.plot(val_accuracy_log,label='val accuracy')
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("model acc")
    plt.grid()
    plt.show()
    
    
    #plot the trainin loss and validation loss
    
    plt.plot(train_loss,label='training loss')
    plt.plot(val_loss,label='validation loss')
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("model loss")
    plt.grid()
    plt.show()
    
    

#visualization using PCA, answer for 1)Vi
def visualize_using_pca(x_val,y_val):
    pca = PCA(n_components=2)
    
    principalComponents = pca.fit_transform(X_val)
    
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    
    yy=pd.DataFrame(data=y_val,columns=['Activities_Types'] )
    finalDf = pd.concat([principalDf, yy], axis = 1)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Using PCA', fontsize = 20)
    
     #0=dws, 1=ups,2=sit, 3=std,4=wlk,5=jog
    targets = [0, 1,2,3,4,5]
    colors = ['g', 'b', 'c','m', 'y', 'r']
    
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Activities_Types'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    targets=['dws','ups','sit','std','wlk','jog']
    ax.legend(targets)
    ax.grid()
    
#visualization using PCA answer for 1Vi
def visualize_using_tsne(x_val,y_val):
    
    
    tsne = TSNE(n_components=2, random_state=0)
    
    principalComponents = tsne.fit_transform(X_val)
    
    #0=dws, 1=ups,2=sit, 3=std,4=wlk,5=jog
    targets = [0, 1,2,3,4,5]

    
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    
    yy=pd.DataFrame(data=y_val,columns=['Activities_Types'] )
    finalDf = pd.concat([principalDf, yy], axis = 1)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Using TSNE', fontsize = 20)
    colors = ['g', 'b', 'c','m', 'y', 'r']
    
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Activities_Types'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    targets=['dws','ups','sit','std','wlk','jog']
    ax.legend(targets)
    ax.grid()
if __name__ == '__main__':
    np.random.seed(42)
    #read the entire dataset
    datasets = pd.read_csv('Data.csv',header=0)
    
    X,y=datasets[datasets.columns.difference(['Activities_Types'])],datasets.loc[:,'Activities_Types']
    
    X,y=X.values,y.values
    
    """
    in the trainin dataset the Activities type starts with 1 and ends at 6
    to make it easy for one hot encoding I subtract one from the y_vale
    and now it becomes 0,1,2,3,4,5 which stands for 0=dws, 1=ups,2=sit, 3=std,4=wlk,5=jog
    later on when we save the test set to a file we will add one to the predicted value
    """
    y=y-1
    
    #do the split training 80% and validation 20%
    
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size = 0.2,random_state=40)
     #build the deep learning network based on the output after we performed a grdi searc
    network = []
    create_the_network()
  
    #variables to record the training accuracy and validation accuracy
    train_accuracy_log = []
    val_accuracy_log = []
    
    train_loss=[]
    val_loss=[]
    
    
    """
    ################################################################################################
    ###                        Hint : switching variable                                         ###
    ###   optimization_option=0 means adam optimizer                                             ###
    ###   optimization_option=1 means SGD                                                        ###
    ###   optimization_option=2 means adagrad                                                    ###
    ##             Thus change the value of this parameter and rerun it to see the difference    ###
    ################################################################################################
            
    """
    
    # I set the defaoult to adam 
    #change this value and rerun the model
    optimization_option=2
    #based on the GridSearch Value I found the following best paramaters for accuracy
    
    if optimization_option==0:
         batch_size=512
    elif optimization_option==1:
         batch_size=128
    else:
        #means adagrad
        batch_size=512
    """
    #start the training procees, show the training progress, display accuracy at each epochs, and finally plot
    #the training accuracy vs validation accuracy
    """
    begin_the_training()
    
    
    """
    ######################################################################################################
    for each class calculate and print the table of Precison, Recall, F1-score, mean average, macro average
    and the weighted average
    ######################################################################################################
    """
    
    precision_recall_f1(X_val,y_val)
    
    
    #answer for question 1 iv Plot rge PCA 
    visualize_using_pca(X_val,y_val)
    visualize_using_tsne(X_val,y_val)
    
    #do the prediction on a test set and save it to the textfile
    
    
    """
    ######################################################################################################
     REMOVE THE COMMENT FROM THE NEXT LINE TO EVALUATE IT ON THE TEST SET
    ######################################################################################################
    """
    #provide the test set file path 
    prediction_on_test_dataset(X_val)
    
    
    """
    ######################################################################################################
     End of the code, THANK YOU VERY MUCH!
    ######################################################################################################
    """
    
    
    