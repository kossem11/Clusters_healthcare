import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv('healthcare_dataset.csv')

x_claster = df[['Age', 'Billing Amount']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x_claster)

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_scaled)


X = df[['Age', 'Admission Type', 'Medical Condition']]
y = df['Billing Amount']
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=24)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R sq:", r2_score(y_test,y_pred))
