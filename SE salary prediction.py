import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle


data=pd.read_csv('salary.csv')
data=data[['Country','EdLevel','YearsCodePro','Employment','ConvertedComp']]
data=data.rename({'ConvertedComp':'Salary'},axis=1)

#---------------------------------------------------------data cleaning ----------------------------------------------------------------------------
data=data.dropna()
#print(data.isnull().sum())

data=data.drop_duplicates()
ch=data.duplicated().any()
#print(ch)

#---------------------------------------------------------to keep just full-time employment ----------------------------------------------------------------------------
data=data[data['Employment']=='Employed full-time']
data=data.drop('Employment',axis=1)
#print(data.info)

#---------------------------------------------------------clean the countries ----------------------------------------------------------------------------

#to know how many people for each country *country : assigned no of people *
v=data['Country'].value_counts()
#print(v)

def shorten_categories(categories, cutoff) :
    categorical_map={}
    for i in range(len(categories)):
      if categories.values[i] >= cutoff:
          categorical_map[categories.index[i]] = categories.index[i]
      else:
          categorical_map[categories.index[i]] = 'others'


    return categorical_map

country_map = shorten_categories(data['Country'].value_counts(),400)
data['Country']=data['Country'].map(country_map)
#print(data['Country'].value_counts())

#---------------------------------------------------------salary with the country----------------------------------------------------------------------------

# fig , ax=plt.subplots(1,1,figsize=(12,7))
# data.boxplot('Salary','Country',ax=ax)
# plt.ylabel('Salary ')
# plt.xticks(rotation=90)

data=data[data['Salary']<=312000]
data=data[data['Salary']>=10000]
data=data[data['Country']!='others']

fig , ax=plt.subplots(1,1,figsize=(12,7))
data.boxplot('Salary','Country',ax=ax)
plt.ylabel('Salary ')
plt.xticks(rotation=90)

#---------------------------------------------------------clean YearsCodePro----------------------------------------------------------------------------


#print(data['YearsCodePro'].unique())

def clean_experience(x):
    if x =='More than 50 years':
        return 50
    if x=='Less than 1 year':
        return 0.5
    return float(x)

data['YearsCodePro']=data['YearsCodePro'].apply(clean_experience)
#print(data['YearsCodePro'].unique())

#---------------------------------------------------------clean EdLevel----------------------------------------------------------------------------

#print(data['EdLevel'].unique())

def clean_EdLevel(x):
    if('Bachelor’s degree' in x ):
        return 'Bachelor’s degree'
    if('Master’s degree' in x ):
        return 'Master’s degree'
    if('Other doctoral degree' in x  or 'Professional degree' in x ):
        return 'post grad'
    return 'less than a bachelor'


data['EdLevel']=data['EdLevel'].apply(clean_EdLevel)

#print(data['EdLevel'].unique())

#---------------------------------------------------------encoding ----------------------------------------------------------------------------

en_education =LabelEncoder()
data['EdLevel']=en_education.fit_transform(data['EdLevel'])
#print(data['EdLevel'].unique())


en_country=LabelEncoder()
data['Country']=en_country.fit_transform(data['Country'])
#print(data['Country'].unique())


#---------------------------------------------------------linear regression----------------------------------------------------------------------------
x=data.drop('Salary',axis=1)
y=data['Salary']

x_train , x_test,y_train , y_test =train_test_split(x,y,test_size=0.2,random_state=0)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)
err=np.sqrt(mean_squared_error(y_test,y_predict))
print("error using lr:" ,err)

#---------------------------------------------------------Decision tree regressor----------------------------------------------------------------------------

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_predict1=dt.predict(x_test)
err1=np.sqrt(mean_squared_error(y_test,y_predict1))
print("error using dt:" ,err1)
#---------------------------------------------------------random forest regressor----------------------------------------------------------------------------
rf=RandomForestRegressor(random_state=0)
rf.fit(x_train,y_train)
y_predict2=rf.predict(x_test)
err2=np.sqrt(mean_squared_error(y_test,y_predict2))
print("error using rf:" ,err2)
#---------------------------------------------------------predict new data----------------------------------------------------------------------------



new_data=np.array([['United States','Master’s degree',15]])

new_data[:,0]=en_country.transform(new_data[:,0])
new_data[:,1]=en_education.transform(new_data[:,1])
new_data=new_data.astype(float)
print("new data entered",new_data)


rf.fit(x,y)
new_data_pred=rf.predict(new_data)
print("predict for new data",new_data_pred)
#---------------------------------------------------------saving the model----------------------------------------------------------------------------

#
# saved={'model':rf,'en_country':en_country,'en_education':en_education}
# with open('save_model.pkl','wb') as file:
#     pickle.dump(saved,file)
#




# with open('save_model.pkl','rb') as file:
#     saved=pickle.load(file)
#
# rf_loaded=saved['model']
# en_country=saved['en_country']
# en_education=saved['en_education']
# y_pre=rf_loaded.predict(new_data)








plt.show()



