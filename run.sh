
echo "🚀 Запуск ML Healthcare App в Docker..."

mkdir -p data

if [ ! -f "data/healthcare_dataset.csv" ]; then
    echo "❌ Файл data/healthcare_dataset.csv не найден!"
    echo "📋 Пожалуйста, поместите ваш CSV файл в папку data/"
    exit 1
fi

echo "🐳 Сборка Docker образа..."
docker-compose build

echo "🚀 Запуск контейнера..."
docker-compose up -d

echo "✅ Приложение запущено!"
echo "🌐 Откройте в браузере: http://localhost:5000"
echo ""
echo "📋 Команды для управления:"
echo "   Остановить: docker-compose down"
echo "   Просмотр логов: docker-compose logs -f"
echo "   Перезапуск: docker-compose restart"