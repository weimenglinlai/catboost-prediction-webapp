# CatBoost Prediction Web App

A CatBoost training pipeline exposed through a lightweight Flask web interface.

## Structure

```
app.py
create_db.py
mode/evaluation_report.txt
mode/feature_columns.json
mode/model_defaults.json
models.py
static/sample_data.csv
static/script.js
static/style.css
templates/batch_predict.html
templates/cities.html
templates/city_detail.html
templates/edit_city.html
templates/history.html
templates/index.html
templates/login.html
templates/new_city.html
templates/register.html
train_catboost.py
```

## Setup

Install dependencies from `requirements.txt` when present:

```bash
pip install -r requirements.txt
```

Data files, model weights and generated outputs are not included; place them under the paths referenced by the configuration before running.
