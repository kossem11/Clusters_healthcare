import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, dbscan
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

class HeartClustering:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.kmeans_model =None
        self.scaler =StandardScaler()

    def preprocessing(self):
        cols = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
        self.df[cols] = self.df[cols].apply(pd.to_numeric, errors = 'coerce')
        self.df =self.df.dropna(subset=cols)

    def heart_clustering(self, n_clusters = 3):
        X = self.df[["Age", "RestingBP", "Cholesterol", "MaxHR"]].values
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster = self.kmeans_model.fit_predict(X_scaled)

        return cluster, X_scaled
    
    def heart_clusters_plot(self, X_scaled, clusters):
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha= 0.6)
        plt.colorbar(scatter)
        plt.title('by Age, RestingBP, Cholesterol, MaxHR')
        
        img =io.BytesIO()
        plt.savefig(img, format = 'png', bbox_inches='tight')
        img.seek(0)
        plot_url =base64.b64encode(img.getvalue()).decode()
        plt.close()

        return plot_url
