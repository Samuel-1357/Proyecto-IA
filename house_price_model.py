"""
House Price Predictor
=====================
Modelo de regresión para predecir el precio de una vivienda
a partir de características como habitaciones, baños, área, etc.

Dataset: California Housing (sklearn)
Tarea:   Regresión supervisada
Modelos: Linear Regression, Random Forest Regressor
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


# ──────────────────────────────────────────────
# 1. CARGA Y PREPARACIÓN DE DATOS
# ──────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Genera un dataset sintético realista de precios de viviendas.

    Se crea con numpy usando distribuciones que replican relaciones
    observadas en datos reales de bienes raíces (sin necesidad de internet).

    Features (X):
        habitaciones        – número de habitaciones  (2–8)
        banos               – número de baños         (1–4)
        area_m2             – área en metros cuadrados (40–400)
        edad_anos           – edad de la casa         (1–60)
        distancia_centro    – distancia al centro     (0.5–30 km)
        estrato             – estrato socioeconómico  (1–6)
        garaje              – tiene garaje            (0 / 1)
        piscina             – tiene piscina           (0 / 1)

    Target (y):
        precio_millones     – precio en millones de COP

    Tamaño: 5,000 muestras
    """
    rng = np.random.default_rng(42)
    n = 5_000

    # ── Features base ──────────────────────────────────────
    estrato          = rng.integers(1, 7, size=n).astype(float)
    habitaciones     = np.clip(rng.normal(3 + estrato * 0.4, 1.0), 1, 10)
    banos            = np.clip(rng.normal(1 + estrato * 0.3, 0.7), 1, 6)
    area_m2          = np.clip(rng.normal(50 + estrato * 30, 30), 20, 500)
    edad_anos        = np.clip(rng.exponential(15), 1, 60)
    distancia_centro = np.clip(rng.exponential(8), 0.5, 40)
    garaje           = (rng.uniform(0, 1) < 0.3 + estrato * 0.1).astype(float)
    piscina          = (rng.uniform(0, 1) < estrato * 0.05).astype(float)

    # ── Precio (función real con ruido) ──────────────────
    precio = (
        50                          # base
        + estrato       * 80        # estrato sube mucho el precio
        + area_m2       *  1.8      # precio por m²
        + habitaciones  * 15        # más habitaciones = más valor
        + banos         * 10        # baños suman valor
        - edad_anos     *  1.5      # casas más viejas valen menos
        - distancia_centro * 4      # lejos del centro = menos valor
        + garaje        * 40        # garaje agrega valor
        + piscina       * 80        # piscina es premium
        + rng.normal(0, 40, n)      # ruido realista
    )
    precio = np.clip(precio, 30, 2000)  # rango realista en millones COP

    X = pd.DataFrame({
        "habitaciones":     np.round(habitaciones).astype(int).astype(float),
        "banos":            np.round(banos).astype(int).astype(float),
        "area_m2":          np.round(area_m2, 1),
        "edad_anos":        np.round(edad_anos).astype(int).astype(float),
        "distancia_centro": np.round(distancia_centro, 2),
        "estrato":          estrato,
        "garaje":           garaje,
        "piscina":          piscina,
    })
    y = pd.Series(np.round(precio, 2), name="precio_millones")

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42):
    """Divide en conjuntos de entrenamiento y prueba."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state)


# ──────────────────────────────────────────────
# 2. PIPELINES DE MODELOS
# ──────────────────────────────────────────────

def build_linear_pipeline() -> Pipeline:
    """
    Pipeline: estandarización → regresión lineal.
    Justificación: sirve como baseline interpretable.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])


def build_random_forest_pipeline(n_estimators: int = 100,
                                  max_depth: int = None,
                                  random_state: int = 42) -> Pipeline:
    """
    Pipeline: estandarización → Random Forest Regressor.
    
    Justificación: captura relaciones no lineales entre features
    y precio sin requerir suposiciones sobre la distribución.
    
    Hiperparámetros:
        n_estimators  – número de árboles (100 por defecto)
        max_depth     – profundidad máxima (None = sin límite)
        random_state  – semilla para reproducibilidad
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1        # usa todos los núcleos disponibles
        ))
    ])


# ──────────────────────────────────────────────
# 3. ENTRENAMIENTO
# ──────────────────────────────────────────────

def train_model(pipeline: Pipeline,
                X_train: pd.DataFrame,
                y_train: pd.Series) -> Pipeline:
    """
    Entrena el pipeline completo (escalado + modelo).
    
    Args:
        pipeline : Pipeline de sklearn sin entrenar
        X_train  : Features de entrenamiento
        y_train  : Target de entrenamiento
    
    Returns:
        Pipeline entrenado
    """
    pipeline.fit(X_train, y_train)
    return pipeline


# ──────────────────────────────────────────────
# 4. EVALUACIÓN
# ──────────────────────────────────────────────

def evaluate_model(pipeline: Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """
    Calcula métricas de evaluación sobre el conjunto de prueba.
    
    Métricas:
        RMSE – Raíz del error cuadrático medio (mismas unidades que y)
        MAE  – Error absoluto medio
        R²   – Coeficiente de determinación (1.0 = predicción perfecta)
    
    Returns:
        dict con métricas y predicciones
    """
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    return {
        "RMSE": round(rmse, 4),
        "MAE":  round(mae, 4),
        "R2":   round(r2, 4),
        "y_pred": y_pred,
        "y_test": y_test.values
    }


def print_metrics(name: str, metrics: dict) -> None:
    """Imprime métricas de forma clara."""
    print(f"\n{'='*45}")
    print(f"  Modelo: {name}")
    print(f"{'='*45}")
    print(f"  RMSE : {metrics['RMSE']:.4f}  (×100k USD)")
    print(f"  MAE  : {metrics['MAE']:.4f}  (×100k USD)")
    print(f"  R²   : {metrics['R2']:.4f}  (0 = pésimo, 1 = perfecto)")
    print(f"{'='*45}")


# ──────────────────────────────────────────────
# 5. GUARDADO / CARGA DE MODELOS
# ──────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str) -> None:
    """Guarda el pipeline entrenado en disco con joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[✓] Modelo guardado en: {path}")


def load_model(path: str) -> Pipeline:
    """Carga un pipeline previamente guardado."""
    return joblib.load(path)


# ──────────────────────────────────────────────
# 6. PREDICCIÓN DE NUEVAS CASAS
# ──────────────────────────────────────────────

FEATURE_NAMES = [
    "habitaciones",
    "banos",
    "area_m2",
    "edad_anos",
    "distancia_centro",
    "estrato",
    "garaje",
    "piscina",
]


def predict_house_price(pipeline: Pipeline,
                        habitaciones: float,
                        banos: float,
                        area_m2: float,
                        edad_anos: float,
                        distancia_centro: float,
                        estrato: float,
                        garaje: float = 0,
                        piscina: float = 0) -> float:
    """
    Predice el precio de una casa dadas sus características.

    Args:
        habitaciones      – número de habitaciones
        banos             – número de baños
        area_m2           – área construida en m²
        edad_anos         – antigüedad de la casa (años)
        distancia_centro  – distancia al centro de la ciudad (km)
        estrato           – estrato socioeconómico (1–6)
        garaje            – 1 si tiene garaje, 0 si no
        piscina           – 1 si tiene piscina, 0 si no

    Returns:
        precio estimado en millones de COP
    """
    sample = pd.DataFrame([[
        habitaciones, banos, area_m2, edad_anos,
        distancia_centro, estrato, garaje, piscina
    ]], columns=FEATURE_NAMES)

    precio = pipeline.predict(sample)[0]
    return max(precio, 0.0)   # los precios no pueden ser negativos
