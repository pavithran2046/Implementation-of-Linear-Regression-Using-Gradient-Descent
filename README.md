# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.
2.Load the dataset into a Pandas DataFrame and preview it using head() and tail().
3.Extract the independent variable X and dependent variable Y from the dataset.
4.Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.
5.Plot the error against the number of epochs to visualize the convergence.
6.Display the final values of m and c, and the error plot.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: pavithran S
RegisterNumber: 212223240113 
*/
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![1](https://github.com/user-attachments/assets/6c163c1b-0dcb-4732-844e-df99dbb111fd)

```
dataset.info()
```
![2](https://github.com/user-attachments/assets/2dbcc63d-f133-4e56-b999-bfc29bc6c68a)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
![3](https://github.com/user-attachments/assets/43d6c9b5-c782-4f87-b1a1-fe86d901ef4e)
```
print(X.shape)
print(Y.shape)
```
![4](https://github.com/user-attachments/assets/f8841ccf-bcfe-43d8-9690-1e8f6fa74ad9)
```
m=0
c=0
L=0.0001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
    Y_pred = m*X +c
    D_m=(-2/n)*sum(X *(Y-Y_pred))
    D_c=(-2/n)*sum(Y -Y_pred)
    m=m-L*D_m
    c=c-L*D_c
    error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
```
![5](https://github.com/user-attachments/assets/092d2541-4b43-4902-b24d-05264b767ac3)

```
plt.plot(range(0,epochs),error)
```

![6](https://github.com/user-attachments/assets/d87319f1-3f6d-40cf-aadf-d722ee451aa0)

## Output:
![7](https://github.com/user-attachments/assets/5d9946af-3feb-4d69-8574-7a63fe9cd436)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
