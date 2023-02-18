#Import library
from library import * 

algo = 'Sigmoid Kernel SVM'

#Load clean data
path = 'C:/Users/Hp/Desktop/THESIS/Data/Main Processing/input data/'
X_clean,y_clean = load_data(path+'df_clean.csv')
print('Data with ignored NaNs loaded')

#Values of Hyperparameters
kernel = ['sigmoid']
degree =  [1]
C = list(np.arange(1,10+1))
gamma = ['scale','auto']
print('\nGrid of parameters defined')

#Multi-threaded Grid Search with Cross Validation
print('\nGrid Search: Started')
thread_list = []
for k in range(len(kernel)):
  for d in range(len(degree)):
    for i in range(len(C)):
      for g in range(len(gamma)):
        t = threading.Thread(target = svm_model,
                             args = [kernel[k],C[i],degree[d],gamma[g],X_clean,y_clean])
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

#extract grid search results
if(os.path.exists('svm_results.csv') == False):
  res_df.to_csv('svm_results.csv')
else:
  res_df.to_csv('svm_results.csv',header = False, mode ='a')
print('Results of Grid Search for ' + algo + ' extracted')