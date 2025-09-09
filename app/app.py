from flask import Flask, render_template, request
from ml_model import HealthcareAnalyzer
import os

app = Flask(__name__)

# Инициализация анализатора
data_path = '../data/healthcare_dataset.csv'
analyzer = HealthcareAnalyzer(data_path)

@app.route('/')
def index():
    # Предобработка данных
    df = analyzer.preprocess_data()
    
    # Кластеризация
    clusters, X_scaled = analyzer.perform_clustering(n_clusters=3)
    cluster_plot = analyzer.create_cluster_plot(X_scaled, clusters)
    
    # Регрессия
    mse, r2, y_test, y_pred = analyzer.perform_regression()
    regression_plot = analyzer.create_regression_plot(y_test, y_pred)
    
    # Статистика
    stats = analyzer.get_statistics()
    
    return render_template('index.html',
                         cluster_plot=cluster_plot,
                         regression_plot=regression_plot,
                         mse=mse,
                         r2=r2,
                         stats=stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)