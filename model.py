import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

df= pd.read_csv('data.csv')

df.drop('Unnamed: 32',axis=1,inplace=True)
df=pd.get_dummies(data=df, drop_first=True)
print(df.head())
x=df.iloc[:,1:-1]
y=df.diagnosis_M
from sklearn import preprocessing
x=preprocessing.StandardScaler().fit(x).transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=10)


#Preparing Logistic Regression
Model=LogisticRegression(C=0.25, random_state=0, solver='liblinear').fit(x_train, y_train)
# Model=LogisticRegression(n_estimators = 120)
# Model.fit(x_train,y_train)


y_predict=Model.predict(x_test)


import pickle
# # Saving model to disk
pickle.dump(Model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_predict)