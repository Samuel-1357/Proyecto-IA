"""
eda_plots.py – Análisis Exploratorio de Datos (EDA)
====================================================
Genera visualizaciones del dataset de precios de viviendas:
  1. Distribución del precio objetivo
  2. Distribución de features clave
  3. Mapa de calor de correlaciones
  4. Scatter: precio vs área y vs estrato

Ejecutar:
    python eda_plots.py
Guarda las figuras en ./plots/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from house_price_model import load_data

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def plot_target_distribution(y: pd.Series) -> None:
    """
    Histograma + líneas de mediana y media del precio.
    Observación esperada: distribución sesgada a la derecha por casas premium.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y, bins=60, edgecolor="white", color="#5C85D6", alpha=0.85)
    ax.axvline(y.median(), color="#D45C5C", linestyle="--", lw=1.5,
               label=f"Mediana: {y.median():.0f}M COP")
    ax.axvline(y.mean(), color="#5CB85C", linestyle="--", lw=1.5,
               label=f"Media: {y.mean():.0f}M COP")
    ax.set_xlabel("Precio (millones COP)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución del precio de las viviendas")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/01_target_distribution.png", dpi=150)
    plt.close(fig)
    print("[✓] plots/01_target_distribution.png")


def plot_feature_distributions(X: pd.DataFrame) -> None:
    """
    Distribuciones de las 4 features más interpretables.
    """
    cols   = ["habitaciones", "banos", "area_m2", "distancia_centro"]
    titles = ["Habitaciones", "Baños", "Área (m²)", "Distancia al centro (km)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for ax, col, title in zip(axes, cols, titles):
        data = X[col].clip(upper=X[col].quantile(0.99))
        ax.hist(data, bins=40, edgecolor="white", color="#7DB87D", alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("Frecuencia")
        p50 = X[col].median()
        ax.axvline(p50, color="#D45C5C", linestyle="--", lw=1.2,
                   label=f"Mediana: {p50:.1f}")
        ax.legend(fontsize=9)

    fig.suptitle("Distribución de features principales", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig("plots/02_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] plots/02_feature_distributions.png")


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Mapa de calor de correlaciones de Pearson.
    Feature más correlacionada con el precio: estrato y área.
    """
    df = X.copy()
    df["precio"] = y.values
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                ax=ax, square=True, cbar_kws={"shrink": 0.75})
    ax.set_title("Correlaciones entre variables (Pearson)", pad=12)
    fig.tight_layout()
    fig.savefig("plots/03_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("[✓] plots/03_correlation_heatmap.png")


def plot_price_vs_area(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Scatter: área vs precio, coloreado por estrato.
    Permite ver la relación positiva área-precio y la separación por estrato.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(X["area_m2"], y,
                    alpha=0.2, s=6, c=X["estrato"], cmap="plasma")
    plt.colorbar(sc, ax=ax, label="Estrato")
    ax.set_xlabel("Área (m²)")
    ax.set_ylabel("Precio (millones COP)")
    ax.set_title("Área vs precio de la vivienda (color = estrato)")
    fig.tight_layout()
    fig.savefig("plots/04_area_vs_price.png", dpi=150)
    plt.close(fig)
    print("[✓] plots/04_area_vs_price.png")


def print_summary(X: pd.DataFrame, y: pd.Series) -> None:
    print("\n" + "="*52)
    print("  RESUMEN DEL DATASET – Viviendas Colombia (sintético)")
    print("="*52)
    print(f"  Muestras       : {len(X):,}")
    print(f"  Features       : {X.shape[1]}")
    print(f"  Precio mínimo  : ${y.min():.0f}M COP")
    print(f"  Precio máximo  : ${y.max():.0f}M COP")
    print(f"  Precio mediano : ${y.median():.0f}M COP")
    print(f"  Valores nulos  : {X.isnull().sum().sum()}")
    for col in ["habitaciones", "area_m2", "distancia_centro"]:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = Q3 - Q1
        n_out = ((X[col] < Q1 - 1.5*iqr) | (X[col] > Q3 + 1.5*iqr)).sum()
        print(f"  Outliers en {col:20s}: {n_out:,}")
    print("="*52)


if __name__ == "__main__":
    print("\nCargando datos...")
    X, y = load_data()
    print_summary(X, y)
    print("\nGenerando gráficas...")
    plot_target_distribution(y)
    plot_feature_distributions(X)
    plot_correlation_heatmap(X, y)
    plot_price_vs_area(X, y)
    print("\n[✓] EDA completo. Revisa la carpeta ./plots/")
