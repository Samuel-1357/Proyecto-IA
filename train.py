"""
train.py – Entrenamiento de modelos
====================================
Entrena dos modelos de regresión y guarda los pipelines en ./models/

    Modelo 1: Regresión Lineal  (baseline)
    Modelo 2: Random Forest     (modelo principal)

Ejecutar:
    python train.py

Salida:
    models/linear_regression.pkl
    models/random_forest.pkl
    plots/05_learning_curve_*.png
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from house_price_model import (
    load_data, split_data,
    build_linear_pipeline, build_random_forest_pipeline,
    train_model, save_model, evaluate_model, print_metrics
)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def plot_learning_curve(pipeline, X_train, y_train, name: str) -> None:
    """
    Curva de aprendizaje (R² en train y validación).
    - Underfitting  : ambas curvas bajas
    - Overfitting   : brecha grande entre train y val
    """
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="r2", cv=3, n_jobs=-1, verbose=0
    )
    tm = train_scores.mean(axis=1); ts = train_scores.std(axis=1)
    vm = val_scores.mean(axis=1);   vs = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_sizes, tm, "o-", color="#5C85D6", label="Entrenamiento (R²)")
    ax.fill_between(train_sizes, tm - ts, tm + ts, alpha=0.15, color="#5C85D6")
    ax.plot(train_sizes, vm, "s-", color="#D45C5C", label="Validación (R²)")
    ax.fill_between(train_sizes, vm - vs, vm + vs, alpha=0.15, color="#D45C5C")
    ax.set_xlabel("Muestras de entrenamiento")
    ax.set_ylabel("R²")
    ax.set_title(f"Curva de aprendizaje – {name}")
    ax.legend(); ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    fname = f"plots/05_learning_curve_{name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"[✓] {fname}")


def main():
    print("\n" + "="*50)
    print("  ENTRENAMIENTO – Predicción de precios de casas")
    print("="*50)

    print("\n[1/4] Cargando y dividiendo datos...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n[2/4] Entrenando Regresión Lineal...")
    t0 = time.time()
    lr = build_linear_pipeline()
    lr = train_model(lr, X_train, y_train)
    print(f"      Tiempo: {time.time()-t0:.2f}s")
    lr_m = evaluate_model(lr, X_test, y_test)
    print_metrics("Regresión Lineal", lr_m)
    save_model(lr, "models/linear_regression.pkl")

    print("\n[3/4] Entrenando Random Forest (100 árboles)...")
    t0 = time.time()
    rf = build_random_forest_pipeline(n_estimators=100, max_depth=15)
    rf = train_model(rf, X_train, y_train)
    print(f"      Tiempo: {time.time()-t0:.2f}s")
    rf_m = evaluate_model(rf, X_test, y_test)
    print_metrics("Random Forest", rf_m)
    save_model(rf, "models/random_forest.pkl")

    print("\n[4/4] Generando curvas de aprendizaje...")
    plot_learning_curve(lr, X_train, y_train, "Regresion Lineal")
    plot_learning_curve(rf, X_train, y_train, "Random Forest")

    print("\n" + "="*50)
    print("  COMPARACIÓN FINAL")
    print("="*50)
    print(f"  {'Métrica':<8}  {'Reg. Lineal':>12}  {'Random Forest':>14}")
    print(f"  {'-'*38}")
    for k in ["RMSE", "MAE", "R2"]:
        print(f"  {k:<8}  {lr_m[k]:>12.4f}  {rf_m[k]:>14.4f}")
    winner = "Random Forest" if rf_m["R2"] > lr_m["R2"] else "Regresión Lineal"
    print(f"\n  Mejor modelo: {winner}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
