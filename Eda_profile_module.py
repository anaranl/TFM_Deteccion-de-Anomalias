"""
eda_profile.py
--------------
Análisis exploratorio de las series mensuales de balance (Fase 2).

Responde las preguntas que condicionan el diseño del modelo:
  1. ¿Cuántas series tienen historia suficiente para el componente temporal?
  2. ¿Qué forma tiene la distribución de movimientos? (define la transformación)
  3. ¿Hay estacionalidad? (los cierres contables tienen dinámica propia)
  4. ¿Cuántas series son demasiado ruidosas para detectar algo?
  5. ¿Con qué frecuencia una cuenta no se mueve? (el cero como información)

Uso en notebook:
    from eda_profile import run_all
    df, profile = run_all()

Uso desde terminal:
    python eda_profile.py

Salidas: outputs/eda/*.png y outputs/eda/series_profile.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import LoadConfig, load_series, series_profile, summarize

OUT = Path("outputs/eda")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# =============================================================================
# 1. Longitud de las series
# =============================================================================

def plot_series_length(profile: pd.DataFrame) -> None:
    """Valida —o refuta— la viabilidad del componente temporal."""
    fig, ax = plt.subplots()
    ax.hist(profile["periods"], bins=range(0, int(profile["periods"].max()) + 2),
            edgecolor="white")
    ax.axvline(24, color="crimson", linestyle="--", linewidth=1.5,
               label="24 períodos (umbral LSTM)")
    ax.axvline(12, color="orange", linestyle="--", linewidth=1.5,
               label="12 períodos (umbral AE simple)")
    ax.set_xlabel("Períodos con datos")
    ax.set_ylabel("Número de cuentas")
    ax.set_title("Distribución de longitud de series — UNV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_series_length.png", dpi=120)
    

    print("\n--- Longitud de series ---")
    for lo, hi, label in [(24, 999, "24+  (LSTM viable)"),
                          (12, 23, "12-23 (AE simple)"),
                          (0, 11, "<12  (solo baseline)")]:
        n = profile["periods"].between(lo, hi).sum()
        print(f"  {label:22s}: {n:4d}  ({100 * n / len(profile):5.1f}%)")


# =============================================================================
# 2. Distribución de movimientos
# =============================================================================

def plot_movement_distribution(df: pd.DataFrame) -> None:
    """
    Los balances contables abarcan varios órdenes de magnitud. Si la escala
    cruda domina, el modelo solo aprenderá de las cuentas grandes: de ahí la
    normalización intra-cuenta y la transformación logarítmica.
    """
    mov = df.loc[df["net_movement"] != 0, "net_movement"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(mov, bins=100, edgecolor="white")
    axes[0].set_title("Movimiento neto — escala cruda")
    axes[0].set_xlabel("USD")

    # log(1+|x|) conservando el signo: comprime la escala sin perder dirección
    signed_log = np.sign(mov) * np.log1p(mov.abs())
    axes[1].hist(signed_log, bins=100, edgecolor="white", color="seagreen")
    axes[1].set_title("Movimiento neto — log con signo")
    axes[1].set_xlabel("log")

    for ax in axes:
        ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(OUT / "02_movement_distribution.png", dpi=120)
    

    print("\n--- Movimiento neto (excluyendo ceros) ---")
    print(f"  n            : {len(mov):,}")
    print(f"  mediana      : {mov.median():,.0f}")
    print(f"  |p1| - |p99| : {mov.abs().quantile(0.01):,.0f} - "
          f"{mov.abs().quantile(0.99):,.0f}")
    print(f"  asimetría    : {mov.skew():,.1f}   (>|2| justifica log)")


# =============================================================================
# 3. Estacionalidad
# =============================================================================

def plot_seasonality(df: pd.DataFrame) -> None:
    """
    Si diciembre se comporta distinto, el modelo lo marcará como anómalo cada
    año. El mes debe entrar como característica explícita, y probablemente los
    cierres necesiten umbral propio.
    """
    d = df.copy()
    d["month"] = d["period_date"].dt.month
    monthly = d.groupby("month").agg(
        median_abs=("net_movement", lambda s: s.abs().median()),
        mean_lines=("line_count", "mean"),
        zero_share=("no_movement", "mean"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    months = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    axes[0].bar(monthly.index, monthly["median_abs"], color="steelblue")
    axes[0].set_title("Movimiento absoluto mediano")
    axes[1].bar(monthly.index, monthly["mean_lines"], color="darkorange")
    axes[1].set_title("Líneas promedio por cuenta")
    axes[2].bar(monthly.index, 100 * monthly["zero_share"], color="grey")
    axes[2].set_title("% de cuentas sin movimiento")

    for ax in axes:
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.set_xlabel("Mes")
    fig.tight_layout()
    fig.savefig(OUT / "03_seasonality.png", dpi=120)
    

    ratio = monthly["median_abs"].max() / monthly["median_abs"].min()
    print("\n--- Estacionalidad ---")
    print(f"  Mes de mayor actividad : {monthly['median_abs'].idxmax()}")
    print(f"  Mes de menor actividad : {monthly['median_abs'].idxmin()}")
    print(f"  Razón máx/mín          : {ratio:.1f}x")
    if ratio > 2:
        print("  -> Estacionalidad relevante: incluir el mes como característica.")


# =============================================================================
# 4. Relación señal-ruido
# =============================================================================

def plot_noise_ratio(profile: pd.DataFrame) -> None:
    """
    Cuando la desviación estándar supera el movimiento medio, la serie tiene
    tan poca señal que solo se detectarán desviaciones extremas. Es el techo
    real de detección, independiente de la arquitectura elegida.
    """
    nr = profile["noise_ratio"].dropna()

    fig, ax = plt.subplots()
    ax.hist(nr.clip(upper=5), bins=50, edgecolor="white", color="indianred")
    ax.axvline(1, color="black", linestyle="--", label="ruido = señal")
    ax.set_xlabel("Desviación estándar / movimiento absoluto medio")
    ax.set_ylabel("Número de cuentas")
    ax.set_title("Relación ruido-señal por serie (recortado en 5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "04_noise_ratio.png", dpi=120)
    

    print("\n--- Relación ruido-señal ---")
    for lo, hi, label in [(0, 0.5, "< 0.5  estable"),
                          (0.5, 1.0, "0.5-1  manejable"),
                          (1.0, 2.0, "1-2    ruidosa"),
                          (2.0, np.inf, "> 2    muy ruidosa")]:
        n = nr.between(lo, hi, inclusive="left").sum()
        print(f"  {label:20s}: {n:4d}  ({100 * n / len(nr):5.1f}%)")


# =============================================================================
# 5. Frecuencia de ceros
# =============================================================================

def plot_zero_share(profile: pd.DataFrame) -> None:
    """
    Un mes sin movimiento no es un hueco: es información. Pero una serie
    mayoritariamente en cero no aporta suficiente para un modelo temporal.
    """
    fig, ax = plt.subplots()
    ax.hist(100 * profile["zero_share"], bins=40, edgecolor="white",
            color="slateblue")
    ax.set_xlabel("% de períodos sin movimiento")
    ax.set_ylabel("Número de cuentas")
    ax.set_title("Frecuencia de períodos sin movimiento")
    fig.tight_layout()
    fig.savefig(OUT / "05_zero_share.png", dpi=120)
    

    mostly_zero = (profile["zero_share"] > 0.5).sum()
    print("\n--- Períodos sin movimiento ---")
    print(f"  Mediana de la proporción de ceros: "
          f"{100 * profile['zero_share'].median():.1f}%")
    print(f"  Series con más del 50% en cero   : {mostly_zero} "
          f"({100 * mostly_zero / len(profile):.1f}%)")


# =============================================================================
# Ejecución
# =============================================================================

def run_all(config: LoadConfig | None = None):
    """
    Ejecutara el EDA completo desde el Notebook. En notebook los gráficos se muestran en línea;
    también quedan guardados en outputs/eda/.
    """
    config = config or LoadConfig()
    df = load_series(config)
    profile = series_profile(df)

    plot_series_length(profile)
    plot_movement_distribution(df)
    plot_seasonality(df)
    plot_noise_ratio(profile)
    plot_zero_share(profile)

    profile.to_csv(OUT / "series_profile.csv")
    summarize(profile, config)

    modelable = profile[profile["periods"] >= config.min_periods]
    print("\n" + "=" * 62)
    print(f"Cuentas modelables (>= {config.min_periods} períodos): {len(modelable)}")
    print(f"Gráficos y perfil guardados en {OUT}/")
    print("=" * 62)

    return df, profile


if __name__ == "__main__":
    run_all()