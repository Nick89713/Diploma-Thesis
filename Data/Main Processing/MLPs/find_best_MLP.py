from library import *

#load results of SVM's grid search
mlp_res = pd.read_csv('mlp_results.csv')
mlp_res = mlp_res.drop(['Unnamed: 0'],axis = 1)
mlp_res = mlp_res.drop_duplicates()
print('Total Results of Grid Search for MLPs loaded')

#Find best combination 
best_row = find_best_combination(mlp_res)
best_param = best_row['Parameters'].to_list()[0]
best_param = ast.literal_eval(best_param)   #convert to dictionary
print('\nBest model found')

#extract best hyperparameters
neurons = best_param['Neurons']
epochs = best_param['Epochs'][0]
batch_size = best_param['Batch Size'][0]
learning_rate = best_param['Learning Rate'][0]
momentum = best_param['Momentum'][0]
print('\nBest hyperparameters extracted')

#Load data where nans were replaced
path = 'C:/Users/Hp/Desktop/THESIS/Data/Main Processing/input data/'
X_mean,y_mean = load_data(path + 'mean_df.csv')
X_mode,y_mode = load_data(path + 'mode_df.csv')
X_median,y_median = load_data(path + 'median_df.csv')
print('\nData with replaced NaNs loaded')

#Test data where nans are replaced
print('\nTrain best model using data with replaced NaNs')
nans_handled = []
mlp_model(neurons,epochs,batch_size,learning_rate,momentum,X_mean,y_mean)
nans_handled.append('Replaced with Mean')
mlp_model(neurons,epochs,batch_size,learning_rate,momentum,X_mode,y_mode)
nans_handled.append('Replaced with Mode')
mlp_model(neurons,epochs,batch_size,learning_rate,momentum,X_median,y_median)
nans_handled.append('Replaced with Median')

#Gather all results in the same dataframe
new_svm_res = pd.DataFrame(list(zip(acc, std,parameters)),
                           columns = ['Mean Accuracy', 'Std of Accuracy', 'Parameters'])
new_svm_res['NaNs Handled'] = nans_handled
common_params = best_row.append(new_svm_res, ignore_index=True)

#Visualize
print('\nPlotting comparisons')
plot_results(common_params,'MLP')