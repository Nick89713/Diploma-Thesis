from library import *
print('Library imported')

algo = 'Random Forest'

#Load clean data
path = 'C:/Users/Hp/Desktop/THESIS/Data/Main Processing/input data/'
X_clean,y_clean = load_data(path+'df_clean.csv')
print('Data with ignored NaNs loaded')

#Values of Hyperparameters
criteria = ['gini', 'entropy']
estimators = [50,100,150,200,250,500]
print('\nGrid of parameters defined')

#Multi-threaded Grid Search with Cross Validation
print('\nGrid Search: Started')
thread_list = []
for i in range(len(criteria)):
  for j in range(len(estimators)):
    t = threading.Thread(target = rf_model,
                        args = [estimators[j], criteria[i],X_clean,y_clean])
    thread_list.append(t)

for j in range(len(thread_list)):
  thread_list[j].start()

for j in range(len(thread_list)):
  thread_list[j].join()
print('\nGrid Search: Ended')

#Keep results of Grid Search with Cross Validation 
res_df = pd.DataFrame(list(zip(acc, std,parameters)),
                           columns = ['Mean Accuracy', 'Std of Accuracy', 'Parameters'])
res_df['Algorithm'] = algo
res_df['NaNs Handled'] = 'Ignored'

#Find best combination 
best_row = find_best_combination(res_df)
best_param = best_row['Parameters'].tolist()[0]
print('\nBest model found')

#Extract best hyperparameters 
trees = best_param['n_estimators'][0]
criterio = best_param['criterion'][0]
print('\nBest hyperparameters extracted')

#Load data where nans were replaced
X_mean,y_mean = load_data(path+'mean_df.csv')
X_mode,y_mode = load_data(path+'mode_df.csv')
X_median,y_median = load_data(path+'median_df.csv')
print('\nData with replaced NaNs loaded')

#Clear lists of acc,std and params & create a list for the way NaNs are handled 
acc.clear()
std.clear()
parameters.clear()
nans_handled = []

#Test data where nans are replaced
print('\nTrain best model using data with replaced NaNs')
rf_model(trees,criterio,X_mean,y_mean)
nans_handled.append('Replaced with Mean')
rf_model(trees,criterio,X_mode,y_mode)
nans_handled.append('Replaced with Mode')
rf_model(trees,criterio,X_median,y_median)
nans_handled.append('Replaced with Median')

#Gather all results in the same dataframe
new_res_df = pd.DataFrame(list(zip(acc, std,parameters)),
                           columns = ['Mean Accuracy', 'Std of Accuracy', 'Parameters'])
new_res_df['Algorithm'] = algo
new_res_df['NaNs Handled'] = nans_handled

#Keep rows with the same hyperparameters
common_params = best_row.append(new_res_df, ignore_index=True)

#Visualize
print('\nPlotting comparisons')
plot_results(common_params,algo)