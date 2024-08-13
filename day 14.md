```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```


```python
df = pd.read_csv("IRIS.csv")
```


```python
df
```


```python
df.head(3)
```


```python
df["species"].unique()
```


```python
sns.pairplot(data=df)
```


```python
sns.pairplot(data=df,hue="species")
```


```python
x = df.iloc[:,:-1]
y = df["species"]
```


```python
x
```


```python
y
```


```python
from sklearn.model_selection import train_test_split
```


```python
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
```

# OVR :


```python
from sklearn.linear_model import LogisticRegression

```


```python
lr = LogisticRegression(multi_class="ovr")
lr.fit(x_train,y_train)
```


```python
lr.score(x_test,y_test)*100
```


```python

```

# MULTINOMIAL :


```python
lr1 = LogisticRegression(multi_class="multinomial")
lr1.fit(x_train,y_train)
```


```python
lr1.score(x_test,y_test)*100
```


```python

```


```python
lr2 = LogisticRegression(multi_class="multinomial")
lr2.fit(x_train,y_train)
```


```python
lr2.score(x_test,y_test)*100
```

# CONFUSION MATRIX :


```python
# CONFUSION MATRIX :
# 1) A confusion matrix is a simple and useful tool for understanding the performance of a classification model,like one used in machine
#    learning or statistics .
# 2) It helps you evaluate how well your model is doing in categorizing things correctly.
# 3) It is also known as the error matrix.
# 4) The matrix consists of predictions result in a summarized form, which has a total number of correct predictions and incorrect predictions.

```


```python
# ACCURACY = TP+TN/N
# ERROR = FN+FP/N
# FALSE NEGATIVE : the model has predicted no,but the actual value was yes ,it is also called as type-2 error.
# FALSE POSITIVE : the model has predicted yes,but the actual value was no.it is called a type-1 error.
```


```python
# CONFUSION MATRIX : (SENSITIVITY,PRECISION,RECALL,F1-SCORE) :
# PRECISION : it helps us to measure the ability to classify positive samples in the model .
# (TP/TP+FP)

# F1-SCORE : it is the harmonic mean of precision and recall.it takes both false positive and false negative into account.therefore,it performs well on an imbalanced dataset.
# FORMULA : F1-SCORE : 2*precision*recall/precision+recall.
```


```python
df = pd.read_csv("data_sets.csv")
```


```python
df
```


```python
df.head(3)
```


```python
x = df.iloc[:,:-1]
y = df["placed"]
```


```python
from sklearn.model_selection import train_test_split
```


```python
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
```


```python

```


```python
from sklearn.linear_model import LogisticRegression

```


```python
lr = LogisticRegression()
lr.fit(x_train,y_train)
```


```python
lr.score(x_test,y_test)*100
```


```python

```


```python
from sklearn.metrics import confusion_matrix,precision_score,f1_score
```


```python
cf = confusion_matrix(y_test,lr.predict(x_test))
cf
```


```python
sns.heatmap(df,annot=True)
```


```python
sns.heatmap(cf,annot=True)
```


```python
precision_score(y_test,lr.predict(x_test))*100
```


```python
from sklearn.metrics import recall_score

```


```python
recall_score(y_test,lr.predict(x_test))*100
```


```python
f1_score(y_test,lr.predict(x_test))*100
```

# IMBALANCED DATASET :

# RANDOM UNDER SAMPLING :


```python
# RANDOM UNDER SAMPLING : we will reduce the majority of the class so that it will have same no of as 
#                         the minority .
```

# RANDOM OVER SAMPLING :


```python
# RANDOM UNDER SAMPLING : we will increase the size of majority is inactive class so the size of 
#                         the minority class is active.
```


```python
df = pd.read_csv("Social_Network_Ads.csv")
```


```python
df.head(3)
```


```python
df["Purchased"].value_counts()
```


```python
x = df.iloc[:,:-1]
y = df["Purchased"]
```


```python
from sklearn.model_selection import train_test_split
```


```python
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
```


```python
from sklearn.linear_model import LogisticRegression

```


```python
lr = LogisticRegression()
lr.fit(x_train,y_train)
```


```python
lr.score(x_test,y_test)*100
```


```python
lr.predict([[19,19000]])
```
