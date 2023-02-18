#Import library
from library import *
print('Library imported')

algo = 'Logistic Regression'

#Load clean data
path = 'C:/Users/Hp/Desktop/THESIS/Data/Main Processing/input data/'
X_clean,y_clean = load_data(path+'df_clean.csv')
print('Data with ignored NaNs loaded')

#Values of Hyperparameters
penalties = ['l1','l2','elasticnet']
C = list(np.arange(1,10+1))
iter = [10000]
solver = ['saga','liblinear']
l1_frac = list(np.arange(0.1,1 + 0.1, 0.1))
print('\nGrid of parameters defined')

#Multi-threaded Grid Search with Cross Validation
print('\nGrid Search: Started')
thread_list = []
for i in range(len(penalties)):
    for j in range(len(C)):
        for k in range(len(solver)):
            for l in range(len(l1_frac)):
                for s in range(len(iter)):
                    t = threading.Thread(target = log_regression_model,
                                         args = [penalties[i],C[j],solver[k],l1_frac[l],iter[s],X_clean,y_clean])
                    thread_list.append(t)

for j in range(len(thread_list)):
  thread_list[j].start()

for j in range(len(thread_list)):
  thread_list[j].join()
print('\nGrid Search: Ended')

#Keep results of Grid Search with Cross Validation 
res_df = pd.DataFrame(list(zip(acc, std,parameters)),
                           columns = ['Mean Accuracy', 'Std of Accuracy', 'Parameters'])
res_df  = res_df[res_df['Parameters'] != {""}]
res_df['Algorithm'] = algo
res_df['NaNs Handled'] = 'Ignored'

#Find best combination 
best_row = find_best_combination(res_df)
best_param = best_row['Parameters'].tolist()[0]
print('\nBest model found')

#Extract best hyperparameters 
penalty = best_param['penalty'][0]
solver = best_param['solver'][0]
l1_ratio = best_param['l1_ratio'][0]
max_iter = best_param['max_iter'][0]
C = best_param['C'][0]
print('\nBest hyperparameters extracted')

#Load data where nans were replaced
X_mean,y_mean = load_data(path + 'mean_df.csv')
X_mode,y_mode = load_data(path + 'mode_df.csv')
X_median,y_median = load_data(path + 'median_df.csv')
print('\nData with replaced NaNs loaded')

#Clear lists of acc,std and params & create a list for the way NaNs are handled 
acc.clear()
std.clear()
parameters.clear()
nans_handled = []

#Test data where nans are replaced
print('\nTrain best model using data with replaced NaNs')
log_regression_model(penalty,C,solver,l1_ratio,max_iter,X_mean,y_mean)
nans_handled.append('Replaced with Mean')
log_regression_model(penalty,C,solver,l1_ratio,max_iter,X_mode,y_mode)
nans_handled.append('Replaced with Mode')
log_regression_model(penalty,C,solver,l1_ratio,max_iter,X_median,y_median)
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