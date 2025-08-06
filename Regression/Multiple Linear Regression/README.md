# Multiple Linear Regression

## Step-1 : Data Loading And Preprocessing

- no need of feature scaling as auto managed by LinearRegression

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")

print(data.head())
print(data.isnull().sum())

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
```

## Step-2 : Encoding non-numerical values

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder= 'passthrough')
X = np.array(ct.fit_transform(X))
```
## Step-3 : Splitting data into tain and test set

```python
from sklearn.model_selection import train_test_split
X_train , X_test, y_train , y_test = train_test_split(X, y , test_size = 0.2 , random_state = 42)
```

## Step-4 : Training the model

 - Simply import from sklearn and then use fit method

 ```python
 from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 ```

## Step-5 : Predicting the results 

- Using predict method

```python
y_pred = np.array(regressor.predict(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```