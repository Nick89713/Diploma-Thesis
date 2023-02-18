#Import library
from library import *

#Load original datasets
df_train = pd.read_csv('pulsar_data_train.csv')
df_test = pd.read_csv('pulsar_data_test.csv')
print('pulsar_data_train.csv loaded')
print('pulsar_data_test.csv loaded')




#Column examination
cols = df_train.columns
num_cols = df_train._get_numeric_data().columns
categoric_cols = list(set(cols) - set(num_cols))
print("\nNumeric columns:")
print(num_cols)
print("\nCategorical columns:")
print(categoric_cols)





#Examine and count NaN values
print("\npulsar_data_train.csv has NaNs in columns: ")
print(df_train.columns[df_train.isna().any()].tolist())
print("pulsar_data_test.csv has NaNs in columns: ")
print(df_test.columns[df_test.isna().any()].tolist())

print("\npulsar_data_train.csv set's NaNs: ")
print(df_train.isnull().sum())
print("\npulsar_data_test.csv set's NaNs: ")
print(df_test.isnull().sum())

NaN_cols = [' Excess kurtosis of the integrated profile',
            ' Standard deviation of the DM-SNR curve',
            ' Skewness of the DM-SNR curve']
print('\nUsing pulsar_data_train.csv from now on')






#Ignoring NaNs in df_train
initial_train_rows = df_train.shape[0]
print('\nInitial number of training rows = ',initial_train_rows)

#df_clean contains the rows of df_train without NaNs
df_clean = df_train.dropna(subset = NaN_cols)
final_number_of_training_rows = df_clean.shape[0]
print('Number of training rows after ignoring NaNs = ', final_number_of_training_rows)
print('Percent of remained rows  after ignoring NaNs = ', str(final_number_of_training_rows/initial_train_rows) + " %")





#Replace NaNs by mean,mode &  median
mean_df = df_train
mode_df = df_train
median_df = df_train

NaN_cols = df_train.columns[df_train.isna().any()].tolist()
for col in NaN_cols:
  mean = df_train[col].mean()
  mode = df_train[col].mode()
  median = df_train[col].median()

  mean_df[col].fillna(value = mean, inplace = True)
  mode_df[col].fillna(value = mode, inplace = True)
  median_df[col].fillna(value = median, inplace = True)

print('\nReplacement of NaNs completed')






#Calculate correlation matrices
sns.heatmap(df_clean.corr(), 
        xticklabels=df_clean.corr().columns,
        yticklabels=df_clean.corr().columns)
plt.savefig('df_clean correlation matrix.png', 
            bbox_inches='tight', pad_inches=0.0)

sns.heatmap(mean_df.corr(), 
        xticklabels=mean_df.corr().columns,
        yticklabels=mean_df.corr().columns)
plt.savefig('mean_df correlation matrix.png', 
            bbox_inches='tight', pad_inches=0.0)

sns.heatmap(mode_df.corr(), 
        xticklabels=mode_df.corr().columns,
        yticklabels=mode_df.corr().columns)
plt.savefig('mode_df correlation matrix.png', 
            bbox_inches='tight', pad_inches=0.0)

sns.heatmap(median_df.corr(), 
        xticklabels=mode_df.corr().columns,
        yticklabels=mode_df.corr().columns)
plt.savefig('median_df correlation matrix.png', 
            bbox_inches='tight', pad_inches=0.0)






#Dimensionality reduction
mean_df = mean_df.drop([' Excess kurtosis of the DM-SNR curve',
                        ' Excess kurtosis of the integrated profile',
                        ' Standard deviation of the DM-SNR curve'],
                        axis = 1)

df_clean = df_clean.drop([' Excess kurtosis of the DM-SNR curve',
                          ' Excess kurtosis of the integrated profile',
                          ' Standard deviation of the DM-SNR curve'],
                          axis = 1)

mode_df = mode_df.drop([' Excess kurtosis of the DM-SNR curve',
                          ' Excess kurtosis of the integrated profile',
                          ' Standard deviation of the DM-SNR curve'],
                            axis = 1)

median_df = median_df.drop([' Excess kurtosis of the DM-SNR curve',
                              ' Excess kurtosis of the integrated profile',
                              ' Standard deviation of the DM-SNR curve'],
                              axis = 1)
print('\nDiamensions reduced')






#Examine percentage of imbalance
#Examine mean,median & mode dataframe, by examining only one of them
non_pulsars = mean_df[mean_df['target_class'] == 0]
pulsars = mean_df[mean_df['target_class'] == 1]

pulsars_num = len(pulsars)
non_pulsars_num = len(non_pulsars)
print('\nDataframes with replaced NaNs: ')
print("Number of Pulsars = ", pulsars_num)
print("Number of Non - Pulsars = ", non_pulsars_num)
pulsars_ratio = pulsars_num / len(mean_df) * 100
non_pulsars_ratio = non_pulsars_num/len(mean_df) * 100
print("Pulsars/Total Observations = ", pulsars_ratio)
print("Non-Pulsars/Total Observations = ", non_pulsars_ratio)
print("Pulsars/Non-Pulsars = " + str(pulsars_num/non_pulsars_num* 100) + " %\n")

#Examine clean df
non_pulsars = df_clean[df_clean['target_class'] == 0]
pulsars = df_clean[df_clean['target_class'] == 1]

pulsars_num = len(pulsars)
non_pulsars_num = len(non_pulsars)
print('\nDataframes with NaNs ignored: ')
print("Number of Pulsars = ", pulsars_num)
print("Number of Non - Pulsars = ", non_pulsars_num)
pulsars_ratio = pulsars_num / len(df_clean) * 100
non_pulsars_ratio = non_pulsars_num/len(df_clean) * 100
print("Pulsars/Total Observations = ", pulsars_ratio)
print("Non-Pulsars/Total Observations = ", non_pulsars_ratio)
print("Pulsars/Non-Pulsars = " + str(pulsars_num/non_pulsars_num* 100) + " %")







#export dataframes in order to use them in the main processing
path = 'C:/Users/Hp/Desktop/THESIS/Data/Main Processing/input data/'
df_clean.to_csv(path + 'df_clean.csv')
#mean_df.to_csv(path+'mean_df.csv')
#mode_df.to_csv(path+'mode_df.csv')
#median_df.to_csv(path+'median_df.csv')
#print("\nDataframes exported as csv files in : 'C:/Users/Hp/Desktop/THESIS/Data/Main Processing/input data/' ")
