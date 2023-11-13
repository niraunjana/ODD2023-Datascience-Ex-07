# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE AND OUTPUT
```
Developed by : Niraunjana Gayathri G R
Reg No.      : 212222230096
```
```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset.csv")

df.columns
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/1c0ec643-72a3-4c42-8257-1cdf5b8d395e)

```
df.shape
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/7742077b-c8dc-4937-9d1c-11c8336697d9)
```
x=df.drop("Survived",1)
y=df['Survived']
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/08b5d065-5811-4756-9ada-09bb53a0ee6b)
```
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df1.columns
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/204bc688-211f-4062-8b39-a350f77a9e3b)
```
df1['Age'].isnull().sum()
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/14cb9457-3c8e-4702-9c2d-efdad194ffac)
```
df1['Age'].fillna(method='ffill')
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/6f37472f-f911-46cb-bf47-432173017a0b)
```
df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/6681dbd6-0c03-4b26-8439-d4ca2bb3303a)
```
feature=SelectKBest(mutual_info_classif,k=3)

df1.columns
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/45b67573-13e3-42ae-aea5-6465b44d4690)
```
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]

df1.columns
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/bbf853a9-7349-4488-9bc6-794657b62153)
```
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/93167074-f1f4-4134-bd6c-4a5ba4a5a8d8)
```
y=y.to_frame()

y.columns
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/08b8789f-26d0-427c-a64c-ce95857029d6)
```
from sklearn.feature_selection import SelectKBest

data=pd.read_csv("/content/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']

x
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/6d8e7ac5-45f5-446a-8885-072fee840018)
```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data[ "Embarked" ]=data ["Embarked"] .astype ("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data[ "Embarked" ]=data ["Embarked"] .cat.codes

data
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/af4cde37-d6d2-4293-9eca-5d65b67495d7)
```
k=5
selector = SelectKBest(score_func=chi2,k=k)
x_new = selector.fit_transform(x,y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_indices = selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features: ")
print(selected_features)
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/f1fe8618-443a-4632-a0f6-98e1f897f09f)
```
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

sfm = SelectFromModel(model, threshold='mean')

sfm.fit(x,y)

selected_feature = x.columns[sfm.get_support()]

print("Selected Features:")
print(selected_feature)
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/672e7aa3-63fc-4064-bac8-d5942c036ca3)
```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()

num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))

rfe.fit(x,y)
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/ac3e3e3f-ebed-4682-a293-e6c042f2544b)
```
selected_features = x.columns[rfe.support_]

print("Selected Features:")
print(selected_feature)
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/2f23df24-d03b-4baa-ba79-cfa88a3b7b5c)
```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x,y)

feature_importances = model.feature_importances_

threshold = 0.15

selected_features = x.columns[feature_importances > threshold]

print("Selected Features:")
print(selected_feature)
```
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex-07/assets/119395610/e718424f-8354-455a-84f9-8a0778f8cbde)

### RESULT:
Thus, the various feature selection techniques have been performed on the given dataset successfully.
