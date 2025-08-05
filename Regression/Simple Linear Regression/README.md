# Simple Linear Regression

- it finds the best fit line  so that sum of perpendicular distance of data points from line is minimum

- general equation -> y = b + dx


## Step-1 : Data Loading And Preprocessing

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")

data.head()
data.isnull().sum()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state = 42)
```

## Step-2 : Training the model

 - Simply import from sklearn and then use fit method

 ```python
 from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 ```

## Step-3 : Predicting the results 

- Using predict method

```python
y_pred = regressor.predict(X_test)
```

## Step-4 : Visualizing results on train set

- Using matplotlib

```python
plt.scatter(X_train , y_train , color = "red")
plt.plot(X_train , regressor.predict(X_train) , color = "blue" )
plt.title("Salary vs Year of Experience (training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
```
## Step-5 : Visualizing results on test set

- Using matplotlib

```python
plt.scatter(X_test , y_test , color = "red")
plt.plot(X_test , y_pred , color = "blue" )
plt.title("Salary vs Year of Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
```