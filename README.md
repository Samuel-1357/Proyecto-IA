House Price Predictor 
Predicción del precio de viviendas mediante regresión supervisada.

Dataset
Dataset sintético generado localmente con NumPy. No requiere descarga externa.

5,000 muestras
8 features: habitaciones, baños, área (m²), edad de la casa, distancia al centro, estrato, garaje, piscina
Target: precio en millones de COP


Requisitos
Python 3.10 o superior.
Instalar dependencias:
pip install -r requirements.txt

Estructura del proyecto
house_price_predictor/
├── house_price_model.py   # Lógica principal: datos, pipelines, predicción
├── eda_plots.py           # Análisis exploratorio y gráficas
├── train.py               # Entrenamiento y guardado de modelos
├── evaluate.py            # Evaluación detallada y ejemplos de predicción
├── requirements.txt       # Dependencias
├── models/                # Pipelines entrenados (generados al correr train.py)
└── plots/                 # Gráficas generadas automáticamente

Cómo usar el proyecto
Sigue los pasos en este orden exacto.
Paso 1 — Instalar dependencias
pip install -r requirements.txt
Paso 2 — Ejecutar el análisis exploratorio
python eda_plots.py
Esto genera 4 gráficas en la carpeta plots/:

Distribución del precio objetivo
Distribución de features principales
Mapa de calor de correlaciones
Scatter de área vs precio

Paso 3 — Entrenar los modelos
python train.py
Esto entrena dos modelos (Regresión Lineal y Random Forest), imprime las métricas en consola, guarda los pipelines entrenados en models/ y genera las curvas de aprendizaje en plots/.
Paso 4 — Evaluar y ver predicciones
python evaluate.py
Esto carga los modelos entrenados, imprime RMSE, MAE y R² sobre el conjunto de prueba, genera gráficas de predichos vs reales, residuos e importancia de features, y muestra 3 ejemplos de predicción con casas hipotéticas.
Nota: el paso 4 requiere haber completado el paso 3 primero. Si se ejecuta evaluate.py sin haber corrido train.py antes, el script mostrará un mensaje de error indicando que los modelos no fueron encontrados.

Usar el modelo para predecir una casa nueva
Abre Python o un notebook y escribe:
pythonfrom house_price_model import load_model, predict_house_price

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
Ajusta los valores de cada parámetro según la casa que quieras evaluar.


AI Usage Statement
Este proyecto fue desarrollado con asistencia de Claude (Anthropic, claude.ai) para generación de código, documentación y depuración. Todo el código fue revisado y validado por el equipo.
