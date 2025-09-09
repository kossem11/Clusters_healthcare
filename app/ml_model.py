import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class HealthcareAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.kmeans_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self):
        """Предобработка данных"""
        numeric_cols = ['Age', 'Billing Amount', 'Room Number']
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna(subset=numeric_cols)
        
        return self.df
    
    def perform_clustering(self, n_clusters=3):
        """Кластеризация K-Means"""
        X = self.df[['Age', 'Billing Amount', 'Room Number']].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        return clusters, X_scaled
    
    def perform_regression(self):
        """Линейная регрессия"""
        X = self.df[['Age', 'Room Number']]
        y = self.df['Billing Amount']
        
        if 'Admission Type' in self.df.columns and 'Medical Condition' in self.df.columns:
            admission_dummies = pd.get_dummies(self.df['Admission Type'], prefix='adm', drop_first=True)
            medical_dummies = pd.get_dummies(self.df['Medical Condition'], prefix='med', drop_first=True)
            X = pd.concat([X, admission_dummies, medical_dummies], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.regression_model = LinearRegression()
        self.regression_model.fit(X_train, y_train)
        y_pred = self.regression_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mse, r2, y_test, y_pred
    
    def create_cluster_plot(self, X_scaled, clusters):
        """Создание графика кластеризации"""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Кластеризация пациентов по Age и Billing Amount')
        plt.xlabel('Age (scaled)')
        plt.ylabel('Billing Amount (scaled)')
        
        # base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    
    def create_regression_plot(self, y_test, y_pred):
        """Создание графика регрессии"""
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
    
    def get_statistics(self):
        """Получение статистики по данным"""
        stats = {
            'total_patients': len(self.df),
            'avg_age': self.df['Age'].mean(),
            'avg_billing': self.df['Billing Amount'].mean(),
            'min_billing': self.df['Billing Amount'].min(),
            'max_billing': self.df['Billing Amount'].max(),
            'medical_conditions': self.df['Medical Condition'].value_counts().to_dict(),
            'admission_types': self.df['Admission Type'].value_counts().to_dict()
        }
        return stats