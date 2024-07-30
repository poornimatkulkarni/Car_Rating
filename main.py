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
#from keras.utils import to_categorical

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

g = sb.countplot(df['Co2_Rating'])
plt.show()
df['Co2_Rating'].value_counts()

#When dataset has different stad scaler it is important to standardize
#and put everything on the same sacle
sc = StandardScaler()
scaled = sc.fit_transform(X)
print(scaled)

#setting values of scaled values of X to X again
X = pd.DataFrame(scaled)
print(X)

#splitting X and Y values into X train,X_test,Y_test and y train with test size 20% and training size 80% of data set
X_train,X_test,Y_train,Y_test = train_test_split(X ,Y , test_size=0.2, random_state= 0)

print("lenth of X train",len(X_train))
print("Print of X test",len(X_test))

print("lenth of Y train",len(Y_train))
print("lenth of Y Test",len(Y_test))

#Model Building
# 1 Logistic Regression
classifier1 = LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000)
classifier1.fit(X_train,Y_train)

print("Logistic Regression Train score is",classifier1.score(X_train,Y_train))
print("Logistic Regression Test score is",classifier1.score(X_test,Y_test))

Y_Pred_lr = classifier1.predict(X_test)

# Confusion Matrix using  logistic regression
cm_lr = confusion_matrix(Y_test,Y_Pred_lr)
print(cm_lr)
print("Accuracy SCore Of Logistic Regression Is",accuracy_score(Y_test,Y_Pred_lr))

sb.heatmap(cm_lr/np.sum(cm_lr),annot=True,fmt='0.2%',cmap='Reds')
plt.xlabel("Predicted values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")

# SVM(Support Vector Machine)
from sklearn.svm import SVC
classifier2 = SVC(kernel='rbf',random_state=0)
classifier2.fit(X_train,Y_train)

print("SVM train score",classifier2.score(X_train,Y_train))
print("SVM test score",classifier2.score(X_test,Y_test))

Y_Pred = classifier2.predict(X_test)

# Making Confusion Matrix for SVM
cm = confusion_matrix(Y_test,Y_Pred)
print(cm)
print("Accuracy score using SVM",accuracy_score(Y_test,Y_Pred))

#To view confusion matrix in Percentage format
sb.heatmap(cm/np.sum(cm),annot=True,fmt='0.2%',cmap='Blues')
plt.xlabel("Predicted values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")

#Generating Classifier Report
print("REPORT FOR LOGISTIC REGRESSION")

print(classification_report(Y_test,Y_Pred_lr,labels=np.unique(Y_Pred_lr)))

print("REPORT FOR SVM")
print(classification_report(Y_test,Y_Pred,labels=np.unique(Y_Pred_lr)))

print("Recall of LR",recall_score(Y_test,Y_Pred_lr,average='weighted'))
print("Precision of LR",precision_score(Y_test,Y_Pred_lr,average='weighted',labels=np.unique(Y_Pred_lr)))
print("F1 Score of LR",f1_score(Y_test,Y_Pred_lr,average='weighted'))
rmse_lr = sqrt(mean_squared_error(Y_test, Y_Pred_lr))
error_lr = mean_absolute_error(Y_test,Y_Pred_lr)
print("Root mean sqaured error LR",rmse_lr)
print("Mean Absolute error LR",error_lr)

print("Recall of SVM",recall_score(Y_test,Y_Pred,average='weighted',labels=np.unique(Y_Pred)))
print("Precision of SVM",precision_score(Y_test,Y_Pred,average='weighted',labels=np.unique(Y_Pred)))
print("F1 Score of SVM",f1_score(Y_test,Y_Pred,average='weighted'   ))
rmse_svm = sqrt(mean_squared_error(Y_test, Y_Pred))
error_svm =  mean_absolute_error(Y_test,Y_Pred)
print("Root mean sqaured error SVM",rmse_svm)
print("Mean Absolute error SVM",error_svm)

"""
Observation
1 Support Vector Machine Classifier and logistic Regression both are are giving 89% accuracy

2 Support Vector Machine Classifier is used for building model
3 It's Supervised Learning Algorithm
4 Linear Kernel is used to build the model
5 Accuracy of 89% achieved using linear SVM
"""

## Building Predictive System
input1 = int(input("Enter year: "))
input2 = float(input("Enter engine size"))
input3 = int(input("Enter number of cylinders: "))
input4 = float(input("Enter fuel consumption in city: "))
input5 = float(input("Enter fuel consumption on highway: "))
input6 = float(input("Enter fuel consumption in combination: "))
input7 = float(input("Enter fuel consumption in mg: "))
input8 = int(input("Enter co2 emission: "))
input9 = int(input("Enter smog rating: "))
input10 = int(input("Enter transmission: "))
input11 = int(input("Enter fuel type: "))
input12 = int(input("Enter vehicle class: "))
#input_data = (2022,2,4,9.1,7,8.2,34,190,5,6,3,12)

input_data = (input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12)
# change input data into array
input_data_array = np.asarray(input_data)

#reshape array
input_data_reshaped = input_data_array.reshape(1,-1)

#standardizing input array
std_data = sc.transform(input_data_reshaped)
print(std_data)

prediction = classifier2.predict(std_data)
#prediction1 = classifier1.predict(std_data)
if prediction <=5 :
    pr = "This Car Can Harm Nature"
else:
    pr = "This Car Can Be A Good Friend Of Nature"

print(prediction,"{}".format(pr))


#print(model.predict([[2022,2,4,12.4,8.9,10.8,26,252,7,12,3,8]]))



#Dumping model into pickle format
db = open('archit_car.pkl','wb')
pickle.dump(classifier2,db)
db.close()
model_car = pickle.load(open('archit_car.pkl','rb'))
