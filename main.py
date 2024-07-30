import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
from math import sqrt
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.svm import SVC
import flask,requests
import jsonify
from flask import Flask
import joblib
from sklearn import svm
#from keras.utils import to_categoricalin

# Data Reading
df = pd.read_csv(r"C:\Users\Poornima\Desktop\MY2022 Fuel Consumption Ratings.csv")
print(df.shape)
print(df.info())
df.head(10)
#df = df.sample(n=len(df))
#df = df.reset_index(drop=True)

print(df.columns)

""""
Attribute Information
Dataset Information Datasets provide model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.

Model: 4WD/4X4 = Four-wheel drive AWD = All-wheel drive FFV = Flexible-fuel vehicle SWB = Short wheelbase LWB = Long wheelbase EWB = Extended wheelbase

Transmission: A = automatic AM = automated manual AS = automatic with select shift AV = continuously variable M = manual 3 – 10 = Number of gears

Fuel type: X = regular gasoline Z = premium gasoline D = diesel E = ethanol (E85) N = natural gas

Fuel consumption: City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km) - the combined rating (55% city, 45% hwy) is shown in L/100 km and in miles per imperial gallon (mpg)

CO2 emissions: the tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving

CO2 rating: the tailpipe emissions of carbon dioxide rated on a scale from 1 (worst) to 10 (best)

Smog rating: the tailpipe emissions of smog-forming pollutants rated on a scale from 1 (worst) to 10 (best)
diesel 2.67
petrol 2.1

"""

# Data Exploration

viz = df[['Cylinders','Engine Size(L)','CO2 Rating', 'Smog Rating','CO2 Emissions(g/km)','Fuel Consumption(Comb (L/100 km))']]
viz.hist(color = 'Blue', figsize = (5, 5))
plt.tight_layout()
plt.show()

print("Null Report",df.isnull().sum())

renamed_columns = {
    'Model Year' : "year",
    'Make' : 'Make',
    'Model':"Model",
    'Vehicle Class': 'vehicle_class',
    'Engine Size(L)': 'engine_size',
    'Cylinders': 'cylinders',
    'Transmission' : 'transmission',
    'Fuel Type': 'fuel_type',
    'Fuel Consumption (City (L/100 km)': 'fuel_cons_city',
    'Fuel Consumption(Hwy (L/100 km))': 'fuel_cons_hwy',
    'Fuel Consumption(Comb (L/100 km))': 'fuel_cons_comb',
    'Fuel Consumption(Comb (mpg))': 'mpgfuel_cons_comb',
    'CO2 Emissions(g/km)': 'co2' ,
    'CO2 Rating':"co2_rating",
     'Smog Rating':'smog_rating'}

df.rename(renamed_columns, axis='columns', inplace=True)
print(df.columns)

le = LabelEncoder()
label1 = le.fit_transform(df['fuel_type'])
label2 = le.fit_transform(df['transmission'])
label3 = le.fit_transform(df['vehicle_class'])

df.drop(['transmission','fuel_type','vehicle_class'],axis=1, inplace = True)
print("Columns after dropping",df.columns)

print("Columns after dropping",df.columns)
#Putting Label Encoding values in respective columns  fuel Type ,transmission,vehicle class
df["transmission"] = label2
df['fuel_type_new'] = label1
df['vehicle_class_new'] = label3

#data frame with deleted column and newly added label encoded values
print(df)

columns_to_move  = df.pop("co2_rating")
df.insert(14,"Co2_Rating",columns_to_move)

df.drop(['Make','Model'],axis=1, inplace = True)

# setting X and Y values
X = df.iloc[:,:-1].values  # all columnsexcept last co2 rating
Y = df["Co2_Rating"]

#splitting X and Y values into X train,X_test,Y_test and y train with test size 20% and training size 80% of data set
X_train,X_test,Y_train,Y_test = train_test_split(X ,Y , test_size=0.2, random_state= 0)

print("lenth of X train",len(X_train))
print("Print of X test",len(X_test))

print("lenth of Y train",len(Y_train))
print("lenth of Y Test",len(Y_test))

#Model Building
# 1 SVM
classifier1 = svm.SVC()
classifier1.fit(X,Y)

#s = pickle.dumps(classifier1)
#clf2 = pickle.loads(s)
#print(clf2.predict([[2022,2,4,19,7,8,34,400,5,6,3,12]]))

joblib.dump(classifier1, 'filename.pkl') 






#print(model.predict([[2022,2,4,12.4,8.9,10.8,26,252,7,12,3,8]]))


