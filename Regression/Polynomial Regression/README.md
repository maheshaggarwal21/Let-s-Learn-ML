# Polynomial Linear Regression

- general equation -> y = a0 + a1* x^1 + a2* x^2 + a3* x^3 + ... + an* x^n

## Step-1 : Data Loading And Preprocessing

- general equation -> y = a0 + a1*x1 + a2*x2 + a3*x3 + ... + an*xn
- no need of feature scaling as auto managed by LinearRegression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Position_Salaries.csv")

print(data.head())
print(data.isnull().sum())

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values
```

## Step-2 : Training The Polynomial Regression Model On Whole Data

- first we have to convert independent variable (X) to X^1 , X^2,...X^n where value of n is specified as degree attribute
- more is the value of n more is the precision but beware of data overfitting
- then we have to apply linear regression on updated values of X

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_mod = PolynomialFeatures(degree=4)
X_poly = poly_mod.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly,y)
```
## Step-3 : Predicitng Results

```python
y_poly = lin_reg_poly.predict(poly_mod.fit_transform([[6.5]]))
```

## Step-4 : Visualizing the results

 ```python
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg_poly.predict(X_poly),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
 ```

