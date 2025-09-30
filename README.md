# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRIYANKA P
RegisterNumber: 212224230212
*/
```
```python

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```
## Output:
### Data Head:
<img width="390" height="265" alt="image" src="https://github.com/user-attachments/assets/9202edb8-eca6-46f4-bba2-975d270ab530" />
### Data Info:
<img width="603" height="237" alt="image" src="https://github.com/user-attachments/assets/6d14eecf-7275-4dce-81fe-f08583badc37" />

### isnull() sum():
<img width="201" height="88" alt="image" src="https://github.com/user-attachments/assets/6d4eb105-52bd-474b-b649-83e045ae8421" />

### Data Head for salary:
<img width="323" height="234" alt="image" src="https://github.com/user-attachments/assets/528f5442-f02b-4cff-ad0e-45baeb18d366" />

### Mean Squared Error :
<img width="239" height="38" alt="image" src="https://github.com/user-attachments/assets/5b336de8-b10d-4ddf-8d9c-a350ddaf8c7a" />

### r2 Value:
<img width="1065" height="41" alt="image" src="https://github.com/user-attachments/assets/93946697-671b-4428-a33c-91392f12eddd" />

### Data prediction :

<img width="311" height="38" alt="image" src="https://github.com/user-attachments/assets/d816b4a7-c1bf-43af-b32f-25e8497f5adf" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
