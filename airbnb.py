import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
import datetime
from xgboost.sklearn import XGBClassifier
import pickle

#import data
df1 = pd.read_csv("sessions.csv")
df2 = pd.read_csv("train_users_2.csv")
df_test = pd.read_csv("test_users.csv")
print('Import of data finished....')

#rename column id to user_id
df2 = df2.rename(columns = {'id':'user_id'})

#join data and fill na values
df_train_all = pd.merge(df1, df2, on='user_id', how='inner')

means = df_train_all['age'].mean()
df_train_all['age'] = df_train_all['age'].fillna(means)


#create labels for prediction
print('Creating labels....')
df_train_labels = df_train_all[['country_destination']]
df_train_all = df_train_all.drop(['user_id', 'country_destination'], axis=1)
temp_numerical_data = df_train_all[['age','signup_flow', 'date_account_created', 'timestamp_first_active','date_first_booking', 'secs_elapsed']]
df_train_all = df_train_all.drop(['age', 'signup_flow','date_account_created', 'date_first_booking','date_first_booking', 'secs_elapsed'], axis=1)
print(df_train_all)

#convert date
def date_to_int(column, df_train_all):
    df_train_all[column] = pd.to_datetime(df_train_all[column])
    month = []
    day = []
    year = []
    for x in df_train_all[column]:
        month.append(x.month)
        day.append(x.day)
        year.append(x.year)
    df_train_all[column+'_month'] = month
    df_train_all[column+'_month'] = df_train_all[column+'_month'].fillna(df_train_all[column+'_month'].mean())
    df_train_all[column+'_day'] = day
    df_train_all[column+'_day'] = df_train_all[column+'_day'].fillna(df_train_all[column+'_day'].mean())
    df_train_all[column+'_year'] = year
    df_train_all[column+'_year'] = df_train_all[column+'_year'].fillna(df_train_all[column+'_year'].mean())
    df_train_all = df_train_all.drop(column, axis=1)

date_to_int('date_account_created', temp_numerical_data)
date_to_int('date_first_booking', temp_numerical_data)
df_train_all = df_train_all.fillna(' ')

#Preprocessing
encoder = LabelEncoder()
# fit transform labels (targets) and data
print('Preprocessing....')
temp_label = encoder.fit_transform(df_train_labels)
print('Preprocessing data')
temp_data =  df_train_all.apply(encoder.fit_transform)

print(temp_data)

#Apply PCA onto LabelEncoder
pca = PCA()
print('Applying PCA....')
# fit transform labels (targets) and data
final_train_data = pca.fit_transform(temp_data)
final_train_label = pca.fit_transform(temp_label)
final_temp_numerical_data = pca.fit_transform(temp_numerical_data)

# combine the data
df1_final_train = final_train_data + final_temp_numerical_data

# parameters for GridSearch
parameters = {
    'learning_rate' : [0.1,0.01,0.001,.05,.005,.0005],
    'n_estimators'  : [100,200,300,400,500],
    'gamma' : [0,.05,1,1.5,10],
    'base_score' :[0.5,0.6,0.7,0.8,0.9]
}


#Training the model
print('Training model....')
model = GridSearch(XGBClassifier(), parameters)
model.fit(df1_final_train, final_train_label)
pickle.dump(model, open('xgbmodel.pkl', 'rb'))
