"""
evaluate.py – Evaluación detallada y ejemplos de predicción
=============================================================
Carga los modelos entrenados y produce:
  - Métricas en test set (RMSE, MAE, R²)
  - Gráfica predichos vs reales
  - Gráfica de residuos
  - Importancia de features (Random Forest)
  - Predicciones de ejemplo

Ejecutar DESPUÉS de train.py:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from house_price_model import (
    load_data, split_data, load_model,
    evaluate_model, print_metrics,
    predict_house_price, FEATURE_NAMES
)

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


def plot_predicted_vs_actual(metrics, name):
    y_test, y_pred = metrics["y_test"], metrics["y_pred"]
    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Predicción perfecta")
    ax.scatter(y_test, y_pred, alpha=0.25, s=8, color="#5C85D6")
    ax.set_xlabel("Precio real (M COP)"); ax.set_ylabel("Precio predicho (M COP)")
    ax.set_title(f"Real vs Predicho – {name}\nR²={metrics['R2']:.3f}")
    ax.legend(); fig.tight_layout()
    fname = f"plots/06_pred_vs_real_{name.lower().replace(' ','_')}.png"
    fig.savefig(fname, dpi=150); plt.close(fig); print(f"[✓] {fname}")


def plot_residuals(metrics, name):
    res = metrics["y_test"] - metrics["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(metrics["y_pred"], res, alpha=0.2, s=6, color="#7DB87D")
    axes[0].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Precio predicho"); axes[0].set_ylabel("Residuo")
    axes[0].set_title(f"Residuos – {name}")
    axes[1].hist(res, bins=50, edgecolor="white", color="#D6855C", alpha=0.85)
    axes[1].axvline(res.mean(), color="red", lw=1.5, linestyle="--",
                    label=f"Media: {res.mean():.2f}")
    axes[1].set_xlabel("Residuo"); axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos"); axes[1].legend()
    fig.tight_layout()
    fname = f"plots/07_residuals_{name.lower().replace(' ','_')}.png"
    fig.savefig(fname, dpi=150); plt.close(fig); print(f"[✓] {fname}")


def plot_feature_importance(rf_pipeline):
    importances = rf_pipeline.named_steps["model"].feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(FEATURE_NAMES)), importances[indices],
           color="#5C85D6", edgecolor="white")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in indices], rotation=35, ha="right")
    ax.set_ylabel("Importancia"); ax.set_title("Importancia de features – Random Forest")
    fig.tight_layout(); fig.savefig("plots/08_feature_importance.png", dpi=150)
    plt.close(fig); print("[✓] plots/08_feature_importance.png")


EXAMPLES = [
    dict(desc="Casa pequeña, estrato 2, lejos del centro",
         habitaciones=2, banos=1, area_m2=55, edad_anos=30,
         distancia_centro=18, estrato=2, garaje=0, piscina=0),
    dict(desc="Apartamento estrato 4, zona semi-céntrica",
         habitaciones=3, banos=2, area_m2=90, edad_anos=12,
         distancia_centro=7, estrato=4, garaje=1, piscina=0),
    dict(desc="Casa de lujo, estrato 6, cerca al centro",
         habitaciones=5, banos=4, area_m2=280, edad_anos=5,
         distancia_centro=3, estrato=6, garaje=1, piscina=1),
]


def run_examples(rf_pipeline):
    print("\n" + "="*55)
    print("  EJEMPLOS DE PREDICCIÓN (Random Forest)")
    print("="*55)
    for ex in EXAMPLES:
        desc = ex["desc"]
        precio = predict_house_price(
            rf_pipeline,
            habitaciones=ex["habitaciones"], banos=ex["banos"],
            area_m2=ex["area_m2"], edad_anos=ex["edad_anos"],
            distancia_centro=ex["distancia_centro"], estrato=ex["estrato"],
            garaje=ex["garaje"], piscina=ex["piscina"]
        )
        print(f"\n  {desc}")
        print(f"  Área: {ex['area_m2']} m² | Estrato: {int(ex['estrato'])} | "
              f"Habitaciones: {ex['habitaciones']} | Distancia: {ex['distancia_centro']} km")
        print(f"  → Precio estimado: ${precio:.0f}M COP")
    print("="*55)


def main():
    print("\n" + "="*50)
    print("  EVALUACIÓN – Predicción de precios de casas")
    print("="*50)

    for path in ("models/linear_regression.pkl", "models/random_forest.pkl"):
        if not os.path.exists(path):
            print(f"\n[!] No se encontró {path}. Ejecuta primero: python train.py\n")
            return

    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)

    print("\nCargando modelos...")
    lr = load_model("models/linear_regression.pkl")
    rf = load_model("models/random_forest.pkl")

    print("\nCalculando métricas...")
    lr_m = evaluate_model(lr, X_test, y_test)
    rf_m = evaluate_model(rf, X_test, y_test)
    print_metrics("Regresión Lineal", lr_m)
    print_metrics("Random Forest", rf_m)

    print("\nGenerando gráficas...")
    plot_predicted_vs_actual(lr_m, "Regresion Lineal")
    plot_predicted_vs_actual(rf_m, "Random Forest")
    plot_residuals(lr_m, "Regresion Lineal")
    plot_residuals(rf_m, "Random Forest")
    plot_feature_importance(rf)

    run_examples(rf)
    print("\n[✓] Evaluación completa. Revisa la carpeta ./plots/\n")


if __name__ == "__main__":
    main()
