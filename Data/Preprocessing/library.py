#This python file contains 
#all  the Python librarues,global variables-objects
#and all user defined functions used in the Thesis.

#------------------------------------Basic Libraries#------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import imblearn
import statistics as st

#---------------------------------Machine Learning Libraries#---------------------------------
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#-----------------------------------Deep Learning Libraries#-----------------------------------
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import SGD
from keras.initializers import HeNormal,GlorotNormal
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

#-------------------------Other Libraries#-------------------------
import ast
import threading
import os

#-----------------------------------------------Global Objects-----------------------------------------------
#a global Lock, used for isolating critical sections between  threads.
lock = threading.Lock()

#a global SMOTE oversampler
oversampler = SMOTE()

#a global Min-Max Scaler
scaler = MinMaxScaler()

#global seed 
np.random.seed(42)

#ideal accuracy
ideal_accuracy = 0.85

#Lists to store the hyperparameters, the mean accuracy and standard deviation of accuracy, of a classifier which performs k-fold cross validation.
parameters = []
acc = []
std = []

#--------------------------------------------User Defined Functions-------------------------------------------
#load_data(): 
#Loads data from a csv file, 
#dropping duplicates and performing SMOTE oversampling. 
#Returns the oversampled features and target classes.
def load_data(filename):
    res_df = pd.read_csv(filename)
    res_df = res_df.drop(['Unnamed: 0'],axis = 1)
    res_df = res_df.drop_duplicates()
    X,y =  oversampler.fit_resample(res_df.loc[:,res_df.columns != 'target_class'],
                                    res_df['target_class'])
    X = X.to_numpy()
    return X,y

#find_best_combionation():
#Finds the best combination of parameters (best model) of a Grid Search,
#Given the mean accuracies and corresponding standard deviations of accuracies,
def find_best_combination(df):   
    acc_list = df['Mean Accuracy'].tolist()
    acc_std_list = df['Std of Accuracy'].tolist()
    param_list = df['Parameters'].tolist()
    NaNs_Handled = df['NaNs Handled'].tolist()
    Algorithm = df['Algorithm'].tolist()
    
    #For each pair of mean accuracy and std of accuracy
    #find the possible min and max accuracy.
    max_accuracy = []
    min_accuracy = []
    for i in range(len(acc_list)):
        max_accuracy.append(acc_list[i] + acc_std_list[i])
        min_accuracy.append(acc_list[i] - acc_std_list[i])
    
    #Keep the range between each min and max accuracy pairs
    #d_range stores the distance between the min and max accuracies.
    #Included holds boolean values, telling if the ideal accuracy belongs to the range.
    #For d_ideal see beelow.
    d_range = []
    d_ideal = []
    included = []

    for i in range(len(max_accuracy)):
        #Calculate the range
        d_range.append(max_accuracy[i] - min_accuracy[i])

        #If the ideal accuracy lies in the range
        #calculate the mean of the range
        #and keep the distance with the ideal accuracy.
        if(max_accuracy[i] > ideal_accuracy and min_accuracy[i] < ideal_accuracy):
            m = (max_accuracy[i] + min_accuracy[i])/2
            d = abs(m - ideal_accuracy)
            d_ideal.append(d)
            included.append(True)
        #If the range lies below the ideal accuracy
        #keep the distance between the ideal and the max accuracy.
        if(max_accuracy[i] < ideal_accuracy):
            d_ideal.append(ideal_accuracy - max_accuracy[i])
            included.append(False)
        #If the range lies above the ideal accuracy
        #keep the distance between the ideal and the min accuracy
        if(min_accuracy[i] > ideal_accuracy):
            d_ideal.append(abs(ideal_accuracy - min_accuracy[i]))
            included.append(False)
    
    #Create a dataframe containing the above calculations
    distance_df = pd.DataFrame({'Mean Accuracy': acc_list,
                                'Std of Accuracy' : acc_std_list,
                                'd_range': d_range,
                                'd_ideal': d_ideal,
                                'Ideal Acc Included': included,
                                'Parameters': param_list,
                                'Algorithm': Algorithm,
                                'NaNs Handled': NaNs_Handled})
    
    #Separate rows into two groups: one group contains the ranges including the ideal accuracy.
    #For the group which does not contain the ideal accuracy,calculate the total rangem as: distance from ideal + range
    #The other group contains the rest of the rows.
    Included = distance_df.loc[distance_df['Ideal Acc Included'] == True]
    Not_Included = distance_df.loc[distance_df['Ideal Acc Included'] == False]

    temp = Not_Included.copy()  #to avoid SettingWithCopyWarning
    temp["d_total"] = temp.d_range + temp.d_ideal
    Not_Included = temp
   
    if(len(Included) != 0):
        most_narrow_intervals = Included[Included['d_range'] == Included['d_range'].min()]
        best_row = most_narrow_intervals[most_narrow_intervals['d_ideal'] == most_narrow_intervals['d_ideal'].min()]
        best_row = best_row.drop(['d_range','d_ideal','Ideal Acc Included'],axis = 1)
    else:
        best_row = Not_Included[Not_Included['d_total'] == Not_Included['d_total'].min()]
        best_row = best_row.drop(['d_range','d_ideal','Ideal Acc Included','d_total'], axis = 1)
         
    return best_row
    
#plot_results():
#Used to show how the replacement of unknown continuous values
#affects the accuracy of an algorithm's specific model
def plot_results(df, algorithm):
    acc_list = df['Mean Accuracy'].tolist()
    std_list = df['Std of Accuracy'].tolist()
    max_acc = []
    min_acc = []
    for i in range(len(acc_list)):
        max_acc.append( acc_list[i] + std_list[i])
        min_acc.append( acc_list[i] - std_list[i] )

    title_str = "Accuracy of " + algorithm
    plt.style.use('dark_background')
    plt.grid( linestyle='-')
    plt.scatter(df['NaNs Handled'].tolist(), acc_list)
    plt.plot(df['NaNs Handled'].tolist() ,max_acc, color = 'orange',linestyle='--',label='maximum accuracy')
    plt.plot(df['NaNs Handled'].tolist(),acc_list,color = 'cyan')
    plt.plot(df['NaNs Handled'].tolist() ,min_acc, color = 'green',linestyle='--', label = 'minimum accuracy')
    plt.xlabel("NaNs")
    plt.ylabel("Accuracy")

    if(ideal_accuracy > min(acc_list) and ideal_accuracy < max(acc_list)):
        plt.axhline(y = ideal_accuracy, color = 'r', linestyle = '-')

    else:
        title_str = title_str + " (overfitting) "
    title_str = title_str + "--- Ideal Accuracy = " + str(ideal_accuracy)
    plt.title(title_str)
    plt.show()











#knn_model(),log_regression_model(),rf_model(),svm_model:
#Functions performing 10-fold cross validation, written for a multi-threaded grid search (for each algorithm).
#These functions are similar in terms of functionality, having very few differences among them.
#For each algorithm, the methodology is the same and is presented in the function knn_model(), due to KNN's simplicity.
#For the other functions, any differece is indicated.
def knn_model(n,X,y):
    #Initialize thread safe model for classification.
    #For knn, use the euclidean distance(minkowski's second norm)
    #and assume all data points are of equal importance.
    knn_model = KNeighborsClassifier(n_neighbors = n,
                                     metric = 'minkowski',
                                     p = 2,
                                     weights = 'uniform')


    #Prepare k-fold cross validation, with same folds per execution.
    #Each fold will have different Train and Test sets
    kf = KFold(n_splits = 10, random_state = 42, shuffle = True)

    #Lists that will store the scaled data
    S_train = []
    S_test = []
    y_train = []
    y_test = []

    #To store the accuracy given in each fold's classification
    cv_acc = []

    #Perform splits. Scale features based on the training set, to avoid data leakage and 
    #isolate scaling so that scaling of train and test feratures is based on the same min and max values.
    #Keep the pairs of scaled train and test data in the lists mentioned above.
    for train_index, test_index in kf.split(X,y):
        X_train = X[train_index]
        X_test = X[test_index]

        lock.acquire()
        scaler.fit(X_train)
        train_scaled = scaler.transform(X_train)
        test_scaled = scaler.transform(X_test)
        lock.release()

        S_train.append(train_scaled)
        S_test.append(test_scaled)
        y_train.append(y[train_index])
        y_test.append(y[test_index])

    #Cross Validate. 
    for i in range(len(S_train)):
        knn_model.fit(S_train[i], y_train[i])
        pred_i = knn_model.predict(S_test[i])
        cv_acc.append( accuracy_score(y_test[i], pred_i) )

    #Get mean accucracy,std of accuracy & parameters of  model.
    acc_m = st.mean(cv_acc)
    acc_sd = st.stdev(cv_acc)
    grid = {'n_neighbors' : [n]}

    #Append results to lists. Also a critical section, due to the 1-1 
    #matching between the mean accuracy and the standard deviation.
    #Notice that some pairs of mean asscuracy and standard deviation of accuracy maybe identical (rare).
    lock.acquire()
    acc.append(acc_m)
    std.append(acc_sd)
    parameters.append(grid)
    lock.release()

def log_regression_model(penalty,c,solver,fraction,iter,X,y):
    #For logistic regression, there is a chance that a combination of parameters is invalid.
    #If so, assume 0 mean accuracy, 0 std of accuracy and an empty combination of parameters.
    #If a cobination is valid, these statistics will change.
    acc_m = 0
    acc_sd = 0
    grid = {""}

    #If penalty is not elasticnet, ignore the l1 ratio.
    if((penalty == 'l1' and (solver == 'liblinear' or solver == 'saga')) or (penalty == 'l2')):
        fraction = None
    #If penalty is elasticnet, take into account the l1 ratio.
    if ( ((penalty == 'l1') and (solver == 'liblinear') or (solver == 'saga')) or (penalty == 'l2') or   ( (penalty == 'elasticnet') and (solver == 'saga'))):
        logreg_model = LogisticRegression(penalty = penalty,
                                        solver = solver,
                                        l1_ratio = fraction,
                                        max_iter = iter,
                                        C = c)
    #In any other case, the combination is invalid and the function returns
    else:
        lock.acquire()
        acc.append(acc_m)
        std.append(acc_sd)
        parameters.append(grid)
        lock.release()

        return None
    #The code below is similar to knn's code
    kf = KFold(n_splits = 10, random_state = 42, shuffle = True)
    S_train = []
    S_test = []
    y_train = []
    y_test = []

    cv_acc = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]

        lock.acquire()
        scaler.fit(X_train)
        train_scaled = scaler.transform(X_train)
        test_scaled = scaler.transform(X_test)
        lock.release()

        S_train.append(train_scaled)
        S_test.append(test_scaled)
        y_train.append(y[train_index])
        y_test.append(y[test_index])

    for i in range(len(S_train)):
        logreg_model.fit(S_train[i], y_train[i])
        pred_i = logreg_model.predict(S_test[i])
        cv_acc.append( accuracy_score(y_test[i], pred_i) )

    acc_m = st.mean(cv_acc)
    acc_sd = st.stdev(cv_acc)
    grid = {
            'penalty' : [penalty],
            'solver' : [solver],
            'l1_ratio' : [fraction],
            'max_iter' : [iter],
            'C' : [c]}

    lock.acquire()
    acc.append(acc_m)
    std.append(acc_sd)
    parameters.append(grid)
    lock.release()

def rf_model(trees,criterio,X,y):
    #Create a random forest classifier. The whole code is similar to knn's
    rf_model = RandomForestClassifier(n_estimators = trees,
                                    criterion = criterio,
                                    bootstrap = False,
                                    max_depth = None,
                                    max_features = None)
                                  

    kf = KFold(n_splits = 10, random_state = 42, shuffle = True)
    S_train = []
    S_test = []
    y_train = []
    y_test = []

    cv_acc = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]

        lock.acquire()
        scaler.fit(X_train)
        train_scaled = scaler.transform(X_train)
        test_scaled = scaler.transform(X_test)
        lock.release()

        S_train.append(train_scaled)
        S_test.append(test_scaled)
        y_train.append(y[train_index])
        y_test.append(y[test_index])

    for i in range(len(S_train)):
        rf_model.fit(S_train[i], y_train[i])
        pred_i = rf_model.predict(S_test[i])
        cv_acc.append( accuracy_score(y_test[i], pred_i) )

    acc_m = st.mean(cv_acc)
    acc_sd = st.stdev(cv_acc)
    grid = {
        'n_estimators' : [trees],
        'criterion' : [criterio]}

    lock.acquire()
    acc.append(acc_m)
    std.append(acc_sd)
    parameters.append(grid)
    lock.release()

def svm_model(kern,c,deg,g,X,y):
    svm_model = SVC(kernel= kern,
                    C = c,
                    degree = deg,
                    gamma = g,
                    coef0 = 0)

    #The code below is similar to knn's
    kf = KFold(n_splits = 10, random_state = 42, shuffle = True)
    S_train = []
    S_test = []
    y_train = []
    y_test = []

    cv_acc = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]

        lock.acquire()
        scaler.fit(X_train)
        train_scaled = scaler.transform(X_train)
        test_scaled = scaler.transform(X_test)
        lock.release()

        S_train.append(train_scaled)
        S_test.append(test_scaled)
        y_train.append(y[train_index])
        y_test.append(y[test_index])

    for i in range(len(S_train)):
        svm_model.fit(S_train[i], y_train[i])
        pred_i = svm_model.predict(S_test[i])
        cv_acc.append( accuracy_score(y_test[i], pred_i) )

    acc_m = st.mean(cv_acc)
    acc_sd = st.stdev(cv_acc)
    grid = {
        'kernel' : [kern],
        'degree' : [deg],
        'gamma' : [g],
        'C' : [c]}

    lock.acquire()
    acc.append(acc_m)
    std.append(acc_sd)
    parameters.append(grid)
    lock.release()











#mlp_model():
#Function performing 10-fold cross validation, written for a sequential (single-threaded) grid search, for simple MLPs.
#See details in function's body.
def mlp_model(neurons,epochs,batch_size,learning_rate,momentum,X,y): 
    #Initialize a multilayer perceptron
    def create_model():
      
        #Initialize optimizer and loss function.
        #Each neural model should have its own loss function, minimized by the optimizer,
        #therefore the optimizer and the loss function have a 1-1 matching,
        #and should be defined in a thread-safe way.
        BCE = BinaryCrossentropy()
        SGDoptimizer = SGD(learning_rate = learning_rate, momentum = momentum)

        #Initialize neural net layers and activation functions. Using the ReLU in all layers except
        #the output layer, which uses the Sigmoid function.
        neural_net = Sequential()
        for i in range(len(neurons)):
            if(i == 0):
                neural_net.add(Dense(neurons[i],
                                    input_dim = X.shape[1],
                                    kernel_initializer = GlorotNormal(),
                                    activation='relu'))
            if(i > 0 and i < len(neurons) - 1):
                neural_net.add(Dense(neurons[i],
                                    kernel_initializer = GlorotNormal(),
                                    activation = 'relu'))
            if( i == len(neurons) - 1):
                neural_net.add(Dense(neurons[i],
                                    kernel_initializer = HeNormal(),
                                    activation = 'sigmoid'))

        #compile: define loss function and optimizer.
        neural_net.compile(loss = BCE, optimizer = SGDoptimizer, metrics = ['accuracy'])

        #return the neural model
        return neural_net
      
    #Wrap the  model. Use a keras classifier to use the model as an sklearn classifier.
    #Also make use of early stoppings. If the loss function remains the same after 5 epochs, then stop fitting (reducing executoin times ).
    MLP_model = KerasClassifier(build_fn = create_model,
                                epochs = epochs,
                                batch_size = batch_size,
                                verbose = 0,
                                callbacks = [EarlyStopping(monitor = 'loss', patience = 5)])

    #the code below is similar to knn's. 
    #No critical sections exist due to the single threaded grid search for MLPs
    kf = KFold(n_splits = 10, random_state = 42, shuffle = True)
    
    S_train = []
    S_test = []
    y_train = []
    y_test = []
    cv_acc = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]

        scaler.fit(X_train)
        train_scaled = scaler.transform(X_train)
        test_scaled = scaler.transform(X_test)
        
        S_train.append(train_scaled)
        S_test.append(test_scaled)
        y_train.append(y[train_index])
        y_test.append(y[test_index])

    for i in range(len(S_train)):
        MLP_model.fit(S_train[i], y_train[i])
        pred_i = MLP_model.predict(S_test[i])
        cv_acc.append( accuracy_score(y_test[i], pred_i) )

    acc_m = st.mean(cv_acc)
    acc_sd = st.stdev(cv_acc)
    grid = {'Neurons' : neurons,
            'Epochs' : [epochs],
            'Batch Size' : [batch_size],
            'Learning Rate' : [learning_rate],
            'Momentum' : [momentum]}

    acc.append(acc_m)
    std.append(acc_sd)
    parameters.append(grid)

