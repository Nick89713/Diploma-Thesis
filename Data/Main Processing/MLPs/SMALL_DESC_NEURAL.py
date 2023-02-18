#Import library
from library import *

algo = 'Small Descending MLP'

#Load clean data
path = 'C:/Users/Hp/Desktop/THESIS/Data/Main Processing/input data/'
X_clean,y_clean = load_data(path+'df_clean.csv')
print('Data with ignored NaNs loaded')

#Values of Hyperparameters
Epochs = [50,100,150]
Momentum = [0.99,0.8]
Learning_rate = [0.1,0.01,0.001]
Batch_size = [32,64]

#structure of network
Neurons = [X_clean.shape[1]]
Hidden_Neurons = np.arange(10, 0, -2).tolist()
Neurons.extend(Hidden_Neurons)
Neurons.append(1)
print('\nGrid of parameters defined')

#Sequencial Grid Search with Cross Validation (it will take hours)
print('\n(Sequential) Grid Search: Started')
for e in range(len(Epochs)):
    for m in range(len(Momentum)):
        for l in range(len(Learning_rate)):
            for b in range(len(Batch_size)):
              print('Examine new model')
              mlp_model(Neurons, Epochs[e], Batch_size[b], Learning_rate[l], Momentum[m] , X_clean,y_clean)
print('\n(Sequential) Grid Search: Ended')

#Keep results of Grid Search with Cross Validation 
res_df = pd.DataFrame(list(zip(acc, std,parameters)),
                           columns = ['Mean Accuracy', 'Std of Accuracy', 'Parameters'])
res_df['Algorithm'] = algo
res_df['NaNs Handled'] = 'Ignored'

#extract grid search results
if(os.path.exists('mlp_results.csv') == False):
  res_df.to_csv('mlp_results.csv')
else:
  res_df.to_csv('mlp_results.csv',header = False, mode ='a')
print('Results of Grid Search for ' + algo + ' extracted')