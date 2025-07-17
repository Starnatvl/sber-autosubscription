# sber-autosubscription
ML проект: Анализ конверсии пользователей сайта СберАвтоподписка. Предсказание вероятности совершения целевых действий на основе веб-аналитики.

## 👥 Команда
- **@natvls** - Team Lead + Data Engineer + EDA Lead
- **@StrikNa** - Data Structure Analyst + Documentation Lead  
- **@olga_vlasova_n** - Data Quality Analyst + Feature Engineering Lead
- **@iliamiuling** - ML Engineer
- **@Maria_Kob** - Business Analyst + QA Engineer
- **@govzol** - UX Analyst + API Developer

## 📁 Структура проекта
```bash
sber-autosubscription/
├── data/ # Данные
│ ├── raw/ # Исходные данные
│ └── processed/ # Обработанные данные
├── notebooks/ # Jupyter notebooks
├── src/ # Исходный код
├── models/ # Обученные модели
├── tests/ # Тесты
├── docs/ # Документация
├── api/ # API приложение
└── requirements.txt # Зависимости
```

## 🚀 Быстрый старт
```bash
# Клонирование репозитория
git clone https://github.com/Starnatvl/sber-autosubscription.git
cd sber-autosubscription

# Установка зависимостей
pip install -r requirements.txt

# Запуск Jupyter
jupyter notebook
```

📅 План работы
- 16 июля: Анализ данных (@StrikNa, @olga_vlasova_n)
- 17-20 июля: Подготовка данных (@natvls, @olga_vlasova_n)
- 20-23 июля: EDA и UX анализ (@natvls, @govzol)
- 24-26 июля: ML разработка (@iliamiuling, @olga_vlasova_n)
- 26-27 июля: API и тестирование (@govzol, @Maria_Kob)
- 27-28 июля: Документация и финализация (@StrikNa)

🎯 Цели проекта
- Создать модель предсказания конверсии (ROC-AUC ~0.65)
- Разработать API для модели
- Подготовить аналитический отчет
- Получить инсайты для улучшения UX сайта

## 📁 Доступ к данным

В этом проекте используются два датасета, которые доступны по следующим ссылкам:

1. **Датасет:** `ga_sessions.csv`  
   - **Описание:** Данные о сессиях пользователей  
   - **Тип:** CSV  
   - **Размер:** 370,1 МБ  
   - **Ссылка для скачивания:**  
   [Скачать с Google Drive](https://drive.google.com/uc?export=download&amp;id=1E-OJ1E_ZliOldsRZgJzL844IV7IlCOqW)  

   Для загрузки в Python используйте:
   ```python
   import gdown
   import pandas as pd
   
   url_sessions = "https://drive.google.com/uc?export=download&amp;id=1E-OJ1E_ZliOldsRZgJzL844IV7IlCOqW"
   output_sessions = "ga_sessions.csv"
   gdown.download(url_sessions, output_sessions, quiet=False)

   df_sessions = pd.read_csv(output_sessions)

2. **Датасет:** `ga_hits.csv`  
   - **Описание:** Данные о событиях пользователей  
   - **Тип:** CSV  
   - **Размер:** 3,97 ГБ  
   - **Ссылка для скачивания:**  
   [Скачать с Google Drive](https://drive.google.com/uc?export=download&amp;id=1y_014uhvDmD55ZlQ_hC7ptsP9jFecyRd)  

   Для загрузки в Python используйте:
   ```python
   import gdown
   import pandas as pd

   url_hits = "https://drive.google.com/uc?export=download&amp;id=1y_014uhvDmD55ZlQ_hC7ptsP9jFecyRd"
   output_hits = "ga_hits.csv"
   gdown.download(url_hits, output_hits, quiet=False)

   df_hits = pd.read_csv(output_hits)

Проект в рамках магистратуры по Data Science
