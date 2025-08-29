# 🌲 Forest Health & Fire Risk — Big Data & ML Project

This repository contains a Jupyter notebook that explores **forest health** and **fire risk** using
data science, machine learning, and geospatial visualization. The work focuses on understanding
how environmental factors (e.g., DBH, canopy cover, soil moisture, weather, slope, elevation, NDVI)
relate to **tree health status** and a **fire-risk index**, and on building predictive models for both.

> Primary notebook: `Big_Data_Project.ipynb`

---

## 🗂️ Dataset
Use the public forest health dataset from Kaggle:

- **Source:** https://www.kaggle.com/datasets/ziya07/forest-health-and-ecological-diversity  

## 🧭 Project Overview

- **Goal:** Analyze environmental drivers of forest health and estimate fire risk for conservation and management.
- **Data:** Two CSV datasets are used in the notebook
  - `forest_health_data.csv` — core environmental features with plot coordinates.
  - `forest_health_data_with_target.csv` — same features plus targets (e.g., `Health_Status` / fire-risk label).
- **Methods:** 
  - EDA & visualization (distributions, correlations, feature relationships)
  - Preprocessing & feature engineering (cleaning, scaling)
  - **Dimensionality reduction** with PCA
  - **Classification** (RandomForest) for health status
  - **Regression** (GradientBoostingRegressor) for fire-risk score
  - **Model interpretation** (feature importances via Yellowbrick)
  - **Geospatial visualization** with Folium (interactive map HTML)
- **Outputs:** evaluation metrics (accuracy, classification report, confusion matrix, RMSE/R²), saved models, and an interactive `forest_health_map.html`.

---

## 🔧 Environment & Setup

**Python version:** 3.9+ recommended

**Install dependencies**
```bash
pip install -r requirements.txt
# or
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick folium joblib jupyter
```

If your original notebook references absolute Windows paths (e.g., `D:/MS IT/...`), update them to the relative repository paths:
```python
data_1 = pd.read_csv("data/forest_health_data.csv")
data_2 = pd.read_csv("data/forest_health_data_with_target.csv")
```

## 🔬 What the Notebook Covers

1. **Import libraries** needed for data handling, ML, and mapping.
2. **Data exploration**: head, info, missing data checks, distributions (e.g., `DBH`, canopy cover), and correlations.
3. **Preprocessing & feature engineering**: cleaning, type conversions, scaling (`StandardScaler`), train/test split.
4. **PCA**: reduce dimensionality to visualize structure in the feature space.
5. **Model training & evaluation**:
   - `RandomForestClassifier` for **Health_Status** (classification report + confusion matrix).
   - `GradientBoostingRegressor` for **fire-risk score** (RMSE & R²).
   - Optional tuning via `RandomizedSearchCV`.
6. **Feature importance** with Yellowbrick to understand which features drive predictions.
7. **Geospatial mapping**: interactive **Folium** map of plot locations with health-status popups; saved to `maps/forest_health_map.html`.
8. **Model persistence**: save trained models to `models/` via `joblib`.

---

## 📊 Key Variables

- **Inputs** (examples): `DBH`, `Tree_Height`, `Canopy_Cover`, `Soil_Moisture`, `Temperature`, `Rainfall`, `Humidity`, `Slope`, `Elevation`, `NDVI`.
- **Targets**:
  - `Health_Status` (0 = healthy, 1 = unhealthy) for classification.
  - `Fire_Risk_Index` (continuous) for regression.

> Your dataset may use slightly different column names—adjust in the notebook as needed.

---

## ✅ Reproducibility

- Set `random_state=42` (or similar) for deterministic splits and model behaviour.
- Document your dataset versions and preprocessing steps.
- Export plots and metrics to `reports/` to track results over time.

---

## 🗺️ Outputs & Artifacts

- **Interactive map:** `maps/forest_health_map.html` (open in a browser).
- **Saved models:** `models/*.joblib` for reuse in apps or further analysis.
- **Figures:** correlation heatmaps, PCA scatter, confusion matrix, feature importance bar charts.

---

## 🚀 Next Steps (Ideas)

- Add **SHAP** or **LIME** for richer model explanations.
- Build a **Streamlit** app to upload data and get predictions + map views.
- Add **data validation** with `pandera` or `pydantic`.
- Introduce **spatial statistics** or neighbourhood features (e.g., spatial lag).

---
