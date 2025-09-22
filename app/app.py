from flask import Flask, render_template, request
from ml_model import HealthcareAnalyzer
from subclass import SubClass
from heart_cluster_model import HeartClustering
import os

app = Flask(__name__)

# Инициализация анализатора
data_path = '../data/healthcare_dataset.csv'
data_path_heart = '../data/heart.csv'
analyzer = HealthcareAnalyzer(data_path)
heart_cluster = HeartClustering(data_path_heart)

@app.route('/')
def index():
    df = analyzer.preprocess_data()
    heart_df = heart_cluster.preprocessing()

    clusters, X_scaled = analyzer.perform_clustering(n_clusters=3)
    cluster_plot = analyzer.create_cluster_plot(X_scaled, clusters)
    
    h_clusters, HX_scaled = heart_cluster.heart_clustering()
    heart_plot = heart_cluster.heart_clusters_plot(HX_scaled, h_clusters)

    mse, r2, y_test, y_pred = analyzer.perform_regression()
    regression_plot = analyzer.create_regression_plot(y_test, y_pred)
    
    stats = analyzer.get_statistics()

    #
    Sub =SubClass({1,4,4,1})
    #
    
    return render_template('index.html',
                         cluster_plot=cluster_plot,
                         heart_plot=heart_plot,
                         regression_plot=regression_plot,
                         mse=mse,
                         r2=r2,
                         stats=stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)