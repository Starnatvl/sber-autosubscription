"""
SberAuto Prediction API

Этот файл содержит:
1. Инструкцию по установке и запуску API.
2. Реализацию FastAPI-приложения.

=== ИНСТРУКЦИЯ ===

1. Подготовка окружения:

   ```bash
   # Скопировать этот файл в пустую папку
   mkdir sber_auto_api && cd sber_auto_api

   # Создать виртуальное окружение
   python3 -m venv venv

   # Активировать
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate.bat    # Windows CMD
   .\venv\Scripts\Activate.ps1 # Windows PowerShell
   ```

2. Установка зависимостей:

   ```bash
   pip install --upgrade pip

   pip install fastapi uvicorn[standard] joblib pandas scikit-learn==1.6.1 xgboost
   ```

3. Подготовка модели:
   - Создайте папку `model_files/` рядом с этим скриптом.
   - Скопируйте в неё `model.joblib` (сохранённый пайплайн).

4. Запуск API:

   ```bash
   uvicorn complete_api:app --reload --host 0.0.0.0 --port 8000
   ```
   - Сервис будет доступен по адресу http://127.0.0.1:8000

5. Тестирование:

   - Swagger UI: http://127.0.0.1:8000/docs
   POST /predict/ принимает JSON с ключевыми признаками и возвращает { "prediction": 0 или 1 }
   Пример JSON:
      {
        "utm_medium": "cpc",
        "month": 7,
        "part_of_day": "evening",
        "city_group": "Moscow",
        "Device": "Apple",
        "utm_source_group": "other_source",
        "keyword_group": "other_keyword",
        "campaign_group": "summer_campaign",
        "hit_number": 12,
        "visit_number": 1
      }
   - Пример curl:
     ```bash
     curl -X POST http://127.0.0.1:8000/predict/ \
          -H "Content-Type: application/json" \
          -d '{
             "utm_medium": "cpc",
             "month": 7,
             "part_of_day": "evening",
             "city_group": "Moscow",
             "Device": "Apple", 
             "utm_source_group": "other_source",
             "keyword_group": "other_keyword",
             "campaign_group": "summer_campaign",
             "hit_number": 12,
             "visit_number": 1
           }'
     ```

=== КОД API ===
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging, traceback
import os

# Обеспечиваем, что модель загружается из папки model_files
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_files", "model.joblib")
model = joblib.load(MODEL_PATH)

# Список признаков, на которых обучена модель
FEATURES = [
    "utm_medium",
    "month",
    "part_of_day",
    "city_group",
    "Device",
    "utm_source_group",
    "keyword_group",
    "campaign_group",
    "hit_number",
    "visit_number",
]
# Числовые признаки
NUMERICAL = ["hit_number", "visit_number"]


class Visit(BaseModel):
    utm_medium: str
    month: int
    part_of_day: str
    city_group: str
    Device: str
    utm_source_group: str
    keyword_group: str
    campaign_group: str
    hit_number: int
    visit_number: int


app = FastAPI(title="SberAuto Prediction API")


@app.get("/")
async def read_root():
    return {"message": "API is running. Use POST /predict/ to get predictions."}


@app.post("/predict/")
async def make_prediction(visit: Visit):
    features = visit.dict()
    try:
        # Формируем DataFrame
        df = pd.DataFrame([{k: features.get(k) for k in FEATURES}])
        # Конвертация числовых
        for col in NUMERICAL:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Проверка на NaN
        nan_cols = df[NUMERICAL].columns[df[NUMERICAL].isna().any()].tolist()
        if nan_cols:
            raise ValueError(f"Invalid or missing numeric fields: {nan_cols}")
        # Предсказание
        pred = model.predict(df)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        logging.error("Prediction error:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
