# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:04:43 2023

@author: Rahul
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Support vector machines\\forestfires.csv")
df

df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.dtypes

# Dropping columns which are not required
df = df.drop(['dayfri', 'daymon', 'daysat', 'daysun', 'daythu','daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 
              'monthfeb','monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov','monthoct','monthsep'], 
             axis = 1)
df

import matplotlib.pyplot as plt
import seaborn as sns
df.size_category.value_counts().plot(kind='bar')

# Checking for which value of area is categorised into large and small by creating crosstab between area and size_category
pd.crosstab(df.area, df.size_category)

# Boxplots
df.boxplot("FFMC",vert=False)
Q1=np.percentile(df["FFMC"],25)
Q3=np.percentile(df["FFMC"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["FFMC"]<LW
df[df["FFMC"]<LW]
df[df["FFMC"]<LW].shape
df["FFMC"]>UW
df[df["FFMC"]>UW]
df[df["FFMC"]>UW].shape
df["FFMC"]=np.where(df["FFMC"]>UW,UW,np.where(df["FFMC"]<LW,LW,df["FFMC"]))

df.boxplot("DMC",vert=False)
Q1=np.percentile(df["DMC"],25)
Q3=np.percentile(df["DMC"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["DMC"]<LW
df[df["DMC"]<LW]
df[df["DMC"]<LW].shape
df["DMC"]>UW
df[df["DMC"]>UW]
df[df["DMC"]>UW].shape
df["DMC"]=np.where(df["DMC"]>UW,UW,np.where(df["DMC"]<LW,LW,df["DMC"]))


df.boxplot("DC",vert=False)
Q1=np.percentile(df["DC"],25)
Q3=np.percentile(df["DC"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["DC"]<LW
df[df["DC"]<LW]
df[df["DC"]<LW].shape
df["DC"]>UW
df[df["DC"]>UW]
df[df["DC"]>UW].shape
df["DC"]=np.where(df["DC"]>UW,UW,np.where(df["DC"]<LW,LW,df["DC"]))


df.boxplot("ISI",vert=False)
Q1=np.percentile(df["ISI"],25)
Q3=np.percentile(df["ISI"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["ISI"]<LW
df[df["ISI"]<LW]
df[df["ISI"]<LW].shape
df["ISI"]>UW
df[df["ISI"]>UW]
df[df["ISI"]>UW].shape
df["ISI"]=np.where(df["ISI"]>UW,UW,np.where(df["ISI"]<LW,LW,df["ISI"]))


df.boxplot("temp",vert=False)
Q1=np.percentile(df["temp"],25)
Q3=np.percentile(df["temp"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["temp"]<LW
df[df["temp"]<LW]
df[df["temp"]<LW].shape
df["temp"]>UW
df[df["temp"]>UW]
df[df["temp"]>UW].shape
df["temp"]=np.where(df["temp"]>UW,UW,np.where(df["temp"]<LW,LW,df["temp"]))


df.boxplot("RH",vert=False)
Q1=np.percentile(df["RH"],25)
Q3=np.percentile(df["RH"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["RH"]<LW
df[df["RH"]<LW]
df[df["RH"]<LW].shape
df["RH"]>UW
df[df["RH"]>UW]
df[df["RH"]>UW].shape
df["RH"]=np.where(df["RH"]>UW,UW,np.where(df["RH"]<LW,LW,df["RH"]))


df.boxplot("wind",vert=False)
Q1=np.percentile(df["wind"],25)
Q3=np.percentile(df["wind"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["wind"]<LW
df[df["wind"]<LW]
df[df["wind"]<LW].shape
df["wind"]>UW
df[df["wind"]>UW]
df[df["wind"]>UW].shape
df["wind"]=np.where(df["wind"]>UW,UW,np.where(df["wind"]<LW,LW,df["wind"]))

# Plotting Month Vs. temp plot
plt.rcParams['figure.figsize'] = [10, 8]
sns.set(style = "darkgrid", font_scale = 1.3)
monthtemp = sns.barplot(x = 'month', y = 'temp', data = df,
                         order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], palette = 'winter');
monthtemp.set(title = "Month Vs Temp Barplot", xlabel = "Months", ylabel = "Temperature");

plt.rcParams['figure.figsize'] = [10, 8]
sns.set(style = "darkgrid", font_scale = 1.3)
day = sns.countplot(df["day"],order = ['sun','mon','tue','wed','thu','fri','sat'])
day.set(title="countplot for days",xlabel='days',ylabel='count')


sns.heatmap(df.corr(), annot=True, cmap="inferno")
ax = plt.gca()
ax.set_title("HeatMap of Features for the Classes")

sns.pairplot(df)
plt.show()

# Encoding month and day features

df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),
                           (1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
df.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
df.head()

# Encoding target variable 'size category'
df.size_category.replace(('small', 'large'), (0, 1), inplace = True)
df.sample(5)

#Standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('size_category',axis=1))
scaled_features=scaler.transform(df.drop('size_category',axis=1))
df_1=pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_1

#data partion
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_1,df["size_category"],test_size=0.3,random_state=(33))

#support Vector mission
svc = SVC()
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

svc = SVC(kernel='rbf',gamma='scale', C=1)
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

svc = SVC(kernel='poly',degree=3,gamma="scale",C=1)
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

clf = SVC()
param_grid = [{'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)

gsv.best_params_ , gsv.best_score_ 

svc = SVC(kernel='linear',C=15,degree=3,gamma=50)
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

#prediction
y_pred_test = logreg.predict(X_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred_test = classifier.predict(X_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))
