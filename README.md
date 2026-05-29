# House Price Predictor 🏠

**Predicción del precio de viviendas mediante regresión supervisada.**

---

## Dataset

Dataset sintético generado localmente con NumPy. No requiere descarga externa.

- 5,000 muestras
- 8 features: habitaciones, baños, área (m²), edad de la casa, distancia al centro, estrato, garaje, piscina
- Target: precio en millones de COP

---

## Requisitos

Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

---

## Estructura del proyecto

```
house_price_predictor/
├── house_price_model.py   # Lógica principal: datos, pipelines, predicción
├── eda_plots.py           # Análisis exploratorio y gráficas
├── train.py               # Entrenamiento y guardado de modelos
├── evaluate.py            # Evaluación detallada y ejemplos de predicción
├── requirements.txt       # Dependencias
├── models/                # Pipelines entrenados (generados al correr train.py)
└── plots/                 # Gráficas generadas automáticamente
```

---

## Cómo usar el proyecto

Sigue los pasos en este orden exacto.

### Paso 1 — Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 2 — Ejecutar el análisis exploratorio

```bash
python eda_plots.py
```

Genera 4 gráficas en `plots/`:

- Distribución del precio objetivo
- Distribución de features principales
- Mapa de calor de correlaciones
- Scatter de área vs precio

### Paso 3 — Entrenar los modelos

```bash
python train.py
```

Entrena Regresión Lineal y Random Forest, imprime métricas en consola, guarda los pipelines en `models/` y genera las curvas de aprendizaje en `plots/`.

### Paso 4 — Evaluar y ver predicciones

```bash
python evaluate.py
```

Carga los modelos entrenados, imprime RMSE, MAE y R² sobre el conjunto de prueba, genera gráficas de predichos vs reales, residuos e importancia de features, y muestra 3 ejemplos de predicción.

> ⚠️ Requiere haber completado el Paso 3 primero. Si se ejecuta sin modelos entrenados, el script mostrará un mensaje de error.

---

## Predecir una casa nueva

```python
from house_price_model import load_model, predict_house_price

pipeline = load_model("models/random_forest.pkl")

precio = predict_house_price(
    pipeline,
    habitaciones=3,
    banos=2,
    area_m2=90,
    edad_anos=10,
    distancia_centro=5,
    estrato=4,
    garaje=1,
    piscina=0
)

print(f"Precio estimado: ${precio:.0f}M COP")
```

Ajusta los parámetros según la casa que quieras evaluar.

---

## Referencias

1. Breiman, L. (2001). *Random Forests.* Machine Learning, 45, 5–32.
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830. — https://scikit-learn.org
3. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

---

## AI Usage Statement

Este proyecto fue desarrollado con asistencia de **Claude** (Anthropic, claude.ai) para generación de código, documentación y depuración. Todo el código fue revisado y validado por el equipo.
