# Data Preprocessing

## `Step-1` : Importing the Data

- Import the data using pandas using either read_csv or read_excel function based on your data file

```bash
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data.csv")
```

## `Step-2` : Separating independent and dependent variables 

- independent variables are those based on those prediction will be made (all except purchased in this case)
- dependent variables are those whose value is to be predicted (purchased in this case)

```bash
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
```


## `Step-3` : Handling Missing Values
 - Deal with the missing values using Simple Imputer from sklearn which replaces missing values with either mean or median according to your choice
 - Fit method calculates the mean / median of input data and sets it to the imputer
 - Transform method replaces the missing values in the data and return a copy of corrected data , it doesn't alter the original data(X)

```bash
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])
```

## `Step-4` : Encoding Non-numerical columns

- basically giving numerical values to non-numerical columns and splitting them for better rendering by model
- using two tools - ColumnTransformer and OneHotEncoder

```bash
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])] , remainder='passthrough' )
X = np.array(ct.fit_transform(X))
```

- now for the purchased column we have to just label the unique values with numerical values (for example yes-1 , no-0) , no column split
- therefore using LabelEncoder

```bash
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y =  np.array(le.fit_transform(y))
```

## `Step-5` : Splitting Data into Training and Test Set

- using train_test_split from sklearn.model_selection
- any value of random state usable , it just ensures we get the same taining and test set whether how many times we run the code
- 0.2 -> 20%

```bash
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)
```
## `Step-6` : Feature Enhancing 

- should be done after splitting to ensure that the model has no info regarding the test set
- fit method applied to only train set while transform applied to both train and test
- this implies that test set is enhanced based on mean calculated from train set

- done so that value of each column ranges between same value
- it ensures each column is given equal importance
- basically two types - normalisation and standardization
- normalization -> value ranges from [0,1]
- standardization -> value ranges from [-3,3]
- standardization is preferred before evergreen

```bash
from sklearn.preprocessing import StandardScaler
ss =  StandardScaler()
ss.fit(X_train[:,3:])
X_train[:,3:] = ss.transform(X_train[:,3:])
X_test[:,3:] = ss.transform(X_test[:,3:])
```
