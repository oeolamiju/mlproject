# End-to-End Machine Learning Project — Student Performance Predictor

Predicting mathematics scores from demographic and prior-performance features, deployed as a live Flask web application on Azure App Service.

**Live demo:** [studentscoreprediction-grh2cgemf8gfayaj.ukwest-01.azurewebsites.net/predictdata](https://studentscoreprediction-grh2cgemf8gfayaj.ukwest-01.azurewebsites.net/predictdata)

> Free-tier Azure App Services spin down after ~20 minutes idle. First request after a cold start takes 30-60 seconds; subsequent requests are near-instant.

---

## What this project is

A portfolio project that takes a machine learning problem end-to-end — from raw data through training, evaluation, packaging, and production deployment. Structured as a modular MLOps pipeline so components (ingestion, transformation, training, prediction) can be swapped or retrained independently, rather than as a single monolithic notebook.

- **Dataset:** Public student performance data (1,000 records)
- **Problem:** Supervised regression on mixed categorical and numerical tabular data
- **Target:** Mathematics score, predicted from seven features — gender, race/ethnicity, parental level of education, lunch type, test preparation course, reading score, writing score
- **Deployed model:** Linear Regression (R² = 0.880 on hold-out test set)

---

## Architecture

The project follows a modular MLOps pattern that separates each stage of the pipeline into its own component. Any single stage can be modified, retrained, or replaced without breaking the others.

```
raw data (notebook/data/stud.csv)
   │
   ▼
DataIngestion              →  artifacts/train.csv, test.csv
   │
   ▼
DataTransformation         →  artifacts/preprocessor.pkl
   │   (SimpleImputer + StandardScaler for numeric features;
   │    SimpleImputer + OneHotEncoder + StandardScaler for categorical)
   │
   ▼
ModelTrainer               →  artifacts/model.pkl
   │   (compares 7 regressors with GridSearchCV;
   │    persists the best-performing model)
   │
   ▼
PredictPipeline
   │   (loads preprocessor + model at inference time)
   │
   ▼
Flask web app (app.py)
   │
   ▼
Azure App Service (UK West)
```

### Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Data ingestion | `src/components/data_ingestion.py` | Reads raw CSV, splits into train/test, persists to `artifacts/` |
| Data transformation | `src/components/data_transformation.py` | Builds a `ColumnTransformer` pipeline for numerical + categorical features |
| Model training | `src/components/model_trainer.py` | Trains and compares 7 regressors, hyperparameter-tunes via GridSearchCV, saves the best |
| Prediction pipeline | `src/pipeline/predict_pipeline.py` | Loads the fitted preprocessor + model, exposes a `predict()` method |
| Web application | `app.py` | Flask app that renders the input form and returns predictions |
| Custom exception | `src/exception.py` | Structured error handling with file + line context |
| Logging | `src/logger.py` | Rotating log files under `logs/` |

---

## Model selection

Seven regression algorithms were compared using GridSearchCV with cross-validation on the training set. Results on the hold-out test set:

| Model | R² (test) |
|-------|-----------|
| Ridge Regression | 0.881 |
| **Linear Regression** *(deployed)* | **0.880** |
| Random Forest Regressor | 0.853 |
| CatBoost Regressor | 0.852 |
| AdaBoost Regressor | 0.847 |
| XGBRegressor | 0.828 |
| Lasso Regression | 0.825 |
| K-Neighbors Regressor | 0.784 |
| Decision Tree Regressor | 0.732 |

**Why Linear Regression was chosen for deployment.** Ridge and Linear scored within 0.001 of one another — a difference below any meaningful signal band. Linear Regression was selected on interpretability grounds: it produces coefficients that can be explained to a non-technical audience, and the marginal accuracy trade-off is negligible. This is a deliberate choice in favour of transparency over a fractional gain in test-set score.

---

## Running locally

```bash
# 1. Clone
git clone https://github.com/oeolamiju/mlproject.git
cd mlproject

# 2. Create a virtual environment (Python 3.10 recommended)
python -m venv venv310
source venv310/bin/activate       # macOS / Linux
# venv310\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the pipeline (produces artifacts/preprocessor.pkl and artifacts/model.pkl)
python src/components/data_ingestion.py

# 5. Run the web app
python app.py
# Open http://127.0.0.1:5000/predictdata
```

---

## Deployment

The application is deployed to **Azure App Service** in the UK West region.

- **Runtime:** Python 3.10 on Linux
- **Deployment method:** GitHub-integrated continuous deployment from `main`
- **Live URL:** [studentscoreprediction-grh2cgemf8gfayaj.ukwest-01.azurewebsites.net/predictdata](https://studentscoreprediction-grh2cgemf8gfayaj.ukwest-01.azurewebsites.net/predictdata)

---

## Tech stack

- **Language:** Python 3.10
- **Modelling:** scikit-learn, XGBoost, CatBoost
- **Web framework:** Flask
- **Front end:** HTML + inline CSS (templates in `templates/`)
- **Deployment:** Azure App Service (Linux, Python runtime)
- **Version control:** Git + GitHub

---

## What this project demonstrates

- **Modular MLOps pattern** — separated concerns for ingestion, transformation, training, and inference; each component swappable without touching the rest
- **Honest model selection** — comparison across nine candidates, deployed the one that best balances accuracy and interpretability rather than just the highest test-set score
- **Engineering discipline** — structured exception handling, rotating logs, reproducible pipeline (the same scripts that trained the deployed model regenerate all artifacts locally)
- **End-to-end delivery** — from raw CSV through to a live, publicly-accessible web endpoint

---

## What I would extend at production scale

For a system serving real users at scale, the next steps would include:

- **Model registry** (MLflow or Azure ML) instead of pickled artifacts in a folder
- **Monitoring and alerting** on prediction distributions, latency, and error rates
- **Automated retraining triggers** based on drift detection or scheduled cadence
- **Containerisation** (Docker) for consistent runtime across environments
- **CI/CD** with automated testing on the transformation and prediction pipelines
- **Input validation and rate limiting** at the API layer
- **A/B testing infrastructure** for controlled rollout of new model versions

---

## Project structure

```
mlproject/
├── app.py                          # Flask web application
├── requirements.txt
├── setup.py
├── artifacts/                      # Trained model, preprocessor, and CSVs (gitignored)
├── notebook/
│   ├── data/stud.csv               # Raw dataset
│   ├── 1_EDA_STUDENT_PERFORMANCE.ipynb
│   └── 2_MODEL_TRAINING.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
└── templates/
    ├── index.html
    └── home.html
```

---

## Author

**Olaniyi (Niyi) Olamiju** — Data & AI professional, Greater Manchester, UK
[LinkedIn](https://www.linkedin.com/in/niyiolamiju/) · [GitHub](https://github.com/oeolamiju)

---

*Built as a portfolio piece to demonstrate end-to-end ML delivery discipline. Comments and pull requests welcome.*
