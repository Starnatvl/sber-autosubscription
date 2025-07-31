# 🚗 СберАвтоподписка: Анализ и прогнозирование конверсии пользователей

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-orange.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ML проект по предсказанию конверсии пользователей на сайте СберАвтоподписка с использованием веб-аналитики и машинного обучения.

## 🎯 Основные результаты

- **🏆 AUC-ROC: 0.81** (превышает целевые 0.65 на 24%)
- **📊 Данные: 15.46M записей** из Google Analytics
- **⚡ API время ответа: <100ms**
- **🎯 Точность предсказания конверсии: 81%**

## 👥 Команда №15

| Участник | Роль | GitHub | Вклад |
|----------|------|--------|-------|
| **Старинцева Наталья** | Team Lead + EDA Lead | [@natvls](https://github.com/Starnatvl) | Координация, разведочный анализ, очистка данных |
| **Власова Ольга** | Data Quality + Feature Engineering | [@olga_vlasova_n](https://github.com/olga_vlasova_n) | Качество данных, создание признаков |
| **Кобзева Мария** | Data Structure + Documentation | [@Maria_Kob](https://github.com/Maria_Kob) | Структура данных, документация |
| **Мюлинг Илья** | ML Engineer | [@iliamiuling](https://github.com/iliamiuling) | Построение и оптимизация ML-моделей |
| **Стрик Наталья** | Business Analyst + QA | [@StrikNa](https://github.com/StrikNa) | Бизнес-логика, тестирование |
| **Халевин Кирилл** | API Developer | [@govzol](https://github.com/govzol) | Разработка API, развертывание |

## 📁 Структура проекта

```
sber-autosubscription/
├── 📂 notebooks/                    # Jupyter notebooks с анализом
│   ├── analysis_report.ipynb        # 📈 Основной EDA отчет (2.3MB)
│   └── ML_model_Miuling_Ilya.ipynb  # 🤖 ML разработка и модели (298KB)
├── 📂 api/                          # FastAPI приложение
│   └── complete_api.py               # 🚀 REST API для модели (5KB)
├── 📂 models/                       # Обученные модели
│   └── model.joblib                  # 💾 XGBoost модель (5.3MB)
├── 📂 data/                         # Данные проекта
│   ├── processed/
│   │   └── df_merged_clean.pkl       # 🗃️ Очищенные данные (935MB)
│   └── raw/                         # Исходные данные (по ссылкам)
├── 📂 docs/                         # Документация и презентации
│   └── presentation.pdf              # 📋 Итоговая презентация (2.5MB)
├── 📄 requirements.txt               # Python зависимости
└── 📖 README.md                     # Этот файл
```
## 🚀 Быстрый старт

### 1. Клонирование и установка
```bash
# Клонирование репозитория
git clone https://github.com/Starnatvl/sber-autosubscription.git
cd sber-autosubscription

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Загрузка данных
```bash
import gdown
import pandas as pd

# Загрузка ga_sessions.csv (370MB)
url_sessions = "https://drive.google.com/uc?export=download&amp;id=1E-OJ1E_ZliOldsRZgJzL844IV7IlCOqW"
gdown.download(url_sessions, "data/raw/ga_sessions.csv", quiet=False)

# Загрузка ga_hits.csv (3.97GB)
url_hits = "https://drive.google.com/uc?export=download&amp;id=1y_014uhvDmD55ZlQ_hC7ptsP9jFecyRd"
gdown.download(url_hits, "data/raw/ga_hits.csv", quiet=False)
```
## 📁 Получение больших файлов данных

Из-за ограничений GitHub некоторые большие файлы не включены в репозиторий:

- `data/processed/df_merged_clean.pkl` (935MB) - обработанные данные

### Варианты получения:
1. **Скачать исходные данные и обработать:**
```python
   # Запустите notebooks/analysis_report.ipynb
   # Файл будет создан автоматически

### 3. Запуск анализа
```bash
# Запуск Jupyter для анализа
jupyter notebook notebooks/analysis_report.ipynb

# Запуск ML разработки
jupyter notebook notebooks/ML_model_Miuling_Ilya.ipynb
```

### 4. Запуск API

```bash
# Запуск FastAPI сервера
cd api
uvicorn complete_api:app --reload --host 0.0.0.0 --port 8000

# API доступно по адресу: http://localhost:8000
# Документация Swagger: http://localhost:8000/docs
```

## 📊 Exploratory Data Analysis (EDA)

### Обработанные данные
- **Объем:** 15.46M записей, 12 признаков
- **Источники:** ga_sessions.csv (1.86M) + ga_hits.csv (15.7M)
- **Период:** Данные веб-аналитики Google Analytics
- **Целевая переменная:** Event_Target (19 типов конверсионных действий)

### Ключевые находки

#### 🔍 Дисбаланс классов
- **Конверсия:** 0.4% (61,473 из 15.46M событий)
- **Подход:** Использование AUC-ROC метрики для корректной оценки

#### 📱 Анализ устройств
- **Мобильные:** 79.3% трафика - критична мобильная оптимизация
- **Десктопы:** 20% - важны для серьезных покупателей  
- **Unknown OS:** 57% - проблемы с трекингом

#### 🌍 География
- **Россия:** 90% трафика (Москва, СПб)
- **Международные:** США/Европа - перспективные рынки

#### ⏰ Временные паттерны
- **Сезонность:** Пик в декабре (+43% к октябрю)
- **Суточная активность:** 12:00-18:00 максимум конверсий
- **Спад:** Июль (сезон отпусков)

## 🤖 ML-модель

### Сравнение алгоритмов

| Модель | AUC-ROC | Время обучения | Особенности |
|--------|---------|----------------|-------------|
| **XGBoost** ⭐ | **0.81** | 15 мин | Лучший результат, оптимизирован |
| LightGBM | 0.766 | 8 мин | Быстрое обучение |
| Нейросети | 0.766 | 25 мин | Глубокое обучение |
| CatBoost | 0.759 | 12 мин | Хорошо с категориями |
| Random Forest | 0.724 | 20 мин | Интерпретируемая |
| Logistic Regression | 0.664 | 2 мин | Baseline модель |

### Оптимизация XGBoost

```python
# Лучшие гиперпараметры (Optuna)
best_params = {
    'n_estimators': 219,
    'max_depth': 14, 
    'learning_rate': 0.12,
    'subsample': 0.84,
    'colsample_bytree': 0.9
}
```

### Метрики качества
- **AUC-ROC:** 0.81 (отлично)
- **Precision:** 0.73 (хорошо для дисбаланса)
- **Recall:** 0.68 (найдено 68% конверсий)
- **F1-Score:** 0.70 (баланс precision/recall)

## 🎯 Интерпретация признаков

### Важность признаков (SHAP Analysis)

| Признак | Важность | Интерпретация |
|---------|----------|---------------|
| **hit_number** | 35.2% | Количество событий в сессии - главный предиктор |
| **visit_number** | 23.1% | Повторные визиты увеличивают конверсию в 2.8 раза |
| **part_of_day_day** | 18.4% | Дневные посещения на 65% результативнее |
| **city_group_Moscow** | 12.8% | Москвичи конвертируются на 45% чаще |
| **Device_desktop** | 10.5% | Десктоп пользователи серьезнее мобильных |

### Бизнес-инсайты

#### 🎯 Критические точки конверсии
- **3-е действие** - точка принятия решения
- **5+ действий** - конверсия в 3.2 раза выше
- **Визиты 2-4** - 67% всех конверсий

#### 📈 Временные паттерны  
- **Обеденное время** (13:00-14:00) - пик активности
- **Дневные часы** - +65% к вечерним визитам
- **Декабрь** - сезон максимальных продаж

## 🚀 API и развертывание

### FastAPI Features

```python
# Основные endpoint'ы
POST /predict          # Предсказание конверсии
GET  /health          # Проверка состояния
GET  /model/info      # Информация о модели
```

### Производительность API
- **Время ответа:** <100ms
- **Пропускная способность:** 150 RPS
- **Память:** 180MB
- **Swagger UI:** Автоматическая документация

### Пример использования

```python
import requests

# Данные для предсказания
data = {
    "hit_number": 5,
    "visit_number": 2,
    "utm_medium": "cpc",
    "month": 12,
    "part_of_day": "day",
    "city_group": "Moscow",
    "Device": "desktop",
    "utm_source_group": "google",
    "keyword_group": "auto_subscription",
    "campaign_group": "brand"
}

# Запрос предсказания
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()

print(f"Вероятность конверсии: {result['probability']:.3f}")
print(f"Предсказание: {'Конверсия' if result['prediction'] else 'Не конверсия'}")
```

## 📋 Презентация и структура

### Итоговая презентация
- **📄 Файл:** `docs/presentation.pdf` (2.5MB)
- **📊 Слайды:** 14 слайдов с ключевыми результатами
- **🎯 Фокус:** Бизнес-ценность и практические рекомендации

### Рекомендации для бизнеса

#### 🔧 UX оптимизация
- Упростить форму заявки до 3 полей
- Добавить прогресс-bar для сложных процессов
- Триггеры вовлечения после 2-го клика
- Быстрый калькулятор без регистрации

#### 📧 Стратегия удержания  
- Email-серия для не конвертировавшихся (5 писем)
- Персонализация на основе просмотренных авто
- Ретаргетинг для визитов 2-4
- Push-уведомления об избранном

#### ⏰ Временная оптимизация
- +40% бюджет рекламы в дневные часы
- Обеденные таргетированные кампании  
- Онлайн поддержка в пиковые часы
- A/B-тест дневных спецпредложений

## 📝 Технические детали

### Стек технологий
- **Python 3.9+** - основной язык
- **Pandas, NumPy** - обработка данных
- **XGBoost** - машинное обучение
- **FastAPI** - веб-API
- **Jupyter** - анализ и разработка
- **SHAP** - интерпретация моделей

### Зависимости
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
fastapi>=0.68.0
uvicorn>=0.15.0
shap>=0.40.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🏆 Результаты и достижения

### Количественные показатели
- ✅ **AUC-ROC 0.81** - превышает цель 0.65 на 24%
- ✅ **15.46M записей** обработано без потерь
- ✅ **API <100ms** время ответа 
- ✅ **19 целевых действий** корректно определены
- ✅ **6 алгоритмов ML** протестированы и сравнены

### Качественные результаты
- 📊 **Полный EDA** с бизнес-инсайтами
- 🤖 **Интерпретируемая ML-модель** 
- 🚀 **Продакшн-готовое API**
- 📋 **Практические рекомендации** для бизнеса
- 📖 **Исчерпывающая документация**

## 📞 Контакты и поддержка

### Команда проекта
- **Вопросы по EDA:** [@natvls](https://github.com/Starnatvl)
- **Вопросы по ML:** [@iliamiuling](https://github.com/iliamiuling)  
- **Вопросы по API:** [@govzol](https://github.com/govzol)
- **Техническая поддержка:** Создайте [Issue](https://github.com/Starnatvl/sber-autosubscription/issues)

### Академический контекст
**🎓 Проект выполнен в рамках:**
- Магистратуры "Науки о данных" НИ ТГУ
- Практики на базе ООО "Скилфэктори" 
- Учебного хакатона (16-28 июля 2025)


