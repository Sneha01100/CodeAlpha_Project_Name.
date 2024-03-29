
import numpy as np
import pandas as pd
import os

d = pd.read_csv("/content/Titanic-Dataset[1].csv")

d.head()

d.shape

d.info()

d.isnull().sum()

d.duplicated().sum()

Survived = d[d["Survived"]==1]
Non_Survived = d[d["Survived"]==0]
outlier = len(Survived)/float(len(Non_Survived))
print(outlier)
print("Survived : {} " .format(len(Survived)))
print("Non_Survived : {} " .format(len(Non_Survived)))

import seaborn as s
s.countplot(x= d["Survived"] , hue = d["Pclass"])

s.countplot(x= d["Survived"] , hue = d["Sex"])

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

labelencoder = LabelEncoder() # Conversion of Categorical values into Numerical values
d['Sex'] = labelencoder.fit_transform(d['Sex'])

d.head()

features = d[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
target = d["Survived"]

d['Age'].fillna(d['Age'].median(), inplace=True)

d.isnull().sum()

a=d[['Pclass', 'Sex']]
b=target

a_train , a_test, b_train, b_test = train_test_split(features,b, test_size= 0.2, random_state= 0)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
a_train_imputed = imputer.fit_transform(a_train)
a_test_imputed = imputer.transform(a_test)

model_taken = RandomForestClassifier()
model_taken.fit(a_train_imputed, b_train)

predictions = model_taken.predict(a_test_imputed)

from sklearn.metrics import accuracy_score, precision_score , recall_score , f1_score
acc = accuracy_score(b_test , predictions)
print("The accuracy is {}".format(acc))

prec = precision_score(b_test , predictions)
print("The precision is {}".format(prec))

rec = recall_score(b_test , predictions)
print("The recall is {}".format(rec))

f1 = f1_score(b_test , predictions)
print("The F1-Score is {}".format(f1))

import joblib
joblib.dump(model_taken,"Titanic_Survival")

z = joblib.load("Titanic_Survival")

prediction  = z.predict([[1,1,0,1,1,1]])
prediction

if prediction==0:
  print("Sinked")
else:
  print("Non Sinked")
