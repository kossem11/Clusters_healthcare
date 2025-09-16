import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

class AcademicStressAnalyther:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.kmeans_model = None
        self.regresion_model = None
        self.scaler = StandardScaler()
    
    def preprocess_data_academic(self):
        numeric_cols = ["Academic pressure from your home", "Rate your academic stress index"]
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors = 'coerce')
        self.df = self.df.dropna(subset=numeric_cols)
        return self.df
    
    def clustering(self, n_clusters = 3):
        X = self.df[["Academic pressure from your home", "Rate your academic stress index"]].values
        x_csaled = self.scaler.fit_transform(X)

        self.kmeans_model = KMeans(n_clusters= n_clusters, random_state= 42, n_init=10)
        clusters = self.kmeans_model.fit_predict(x_csaled)

        return clusters, x_csaled

    def regression(self):
        X = self.df[["Academic pressure from your home", "Rate your academic stress index"]]
        Y = self.df['What would you rate the academic  competition in your student life']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.regresion_model = LinearRegression()
        self.regresion_model.fit(X_train, X_test)
        y_pred = self.regresion_model.predict(X_test)

        mse = mean_squared_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)

        return mse, r2, Y_test, y_pred
    
    def create_regression_plot(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Предсказание vs Реальные значения (Billing Amount)')
        plt.xlabel('Реальные значения')
        plt.ylabel('Предсказания')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
