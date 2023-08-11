import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loc = "/content/SALES.txt"
df = pd.read_csv(loc, sep="\s+", header=None)

print(df.shape)

df.head()

df.columns = ["Sales", "Advertising"]

df.head()

df.describe()

X = df['Sales'].values
y = df['Advertising'].values

plt.scatter(X, y, color='blue', label="Scatter Plot")
plt.title("Relationship between Sales and Advertising")
plt.xlabel("Sales")
plt.ylabel("Advertising")
plt.show()

X.shape
y.shape
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
X.shape
y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

y_pred

plt.scatter(X, y, color='blue', label="Scatter Plot")
plt.plot(X_test, y_pred, color="red", linewidth=3, label="Lm Line")
plt.title("Relationship between Sales and Advertising")
plt.xlabel("Sales")
plt.ylabel("Advertising")
plt.show()