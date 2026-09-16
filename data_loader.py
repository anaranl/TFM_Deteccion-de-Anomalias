"""
data_loader.py
--------------
loading the monthly balance series to unsupervised analisis
(phase 2, version 1: just UNV).

supported by module `db.py` of the project for the connection.
to use in notebook:

    from data_loader import load_series, series_profile, to_matrix

    df = load_series()
    profile = series_profile(df)
    matrix = to_matrix(df, accounts=profile[profile.periods >= 24].index)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from db import run_query


# =============================================================================
# Configuration and series 
# =============================================================================

@dataclass
class LoadConfig:
    """
    Parmeters of version 1.

    the exclusions are apply here and not in the SQL for the extraction phase
    including like dic-2022 in the future will be change configuration, not pipeline.
    """
    env: str = "PROD" 

    ou_group: str = "UNV"

    #   41.694 filas en 93 cuentas, frente a ~243.000 y ~240 de un mes normal.
    start_date: str = "2023-01-01"
    excluded_periods: list[str] = field(default_factory=list)

    # El mes en curso está incompleto y distorsiona medias y desviaciones.
    exclude_current_month: bool = True

    # Longitud mínima de serie para el modelado temporal.
    min_periods: int = 24


QUERY_SERIES = """
SELECT  account,
        ou_group,
        period_name,
        period_date,
        net_movement,
        total_debits,
        total_credits,
        line_count,
        active_ou_count,
        no_movement,
        running_balance
FROM    notif.vw_ML_L2_CompleteSeries
WHERE   ou_group = '{ou_group}'
ORDER BY account, period_date
"""


# =============================================================================
# loading
# =============================================================================

def load_series(config: LoadConfig | None = None, verbose: bool = True) -> pd.DataFrame:
    config = config or LoadConfig()

    df = run_query(QUERY_SERIES.format(ou_group=config.ou_group), env=config.env)
 
    df["period_date"] = pd.to_datetime(df["period_date"])
    n_raw = len(df)
 
    if config.start_date:
        df = df[df["period_date"] >= pd.Timestamp(config.start_date)]
 
    if config.excluded_periods:
        df = df[~df["period_name"].isin(config.excluded_periods)]
 
    if config.exclude_current_month:
        last = df["period_date"].max()
        df = df[df["period_date"] < last]
 
    df = df.sort_values(["account", "period_date"]).reset_index(drop=True)
 
    if verbose:
        print(f"Filas: {len(df):,} de {df['account'].nunique()} cuentas "
              f"({n_raw - len(df):,} excluidas)")
        print(f"Rango: {df['period_date'].min():%Y-%m} a {df['period_date'].max():%Y-%m}")
 
    return df


# =============================================================================
# Perfil por serie
# =============================================================================

def series_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por cuenta con las métricas que deciden qué modelo aplica.

    `noise_ratio` es la métrica crítica: desviación estándar sobre movimiento
    absoluto medio. Por encima de 1, la serie tiene tan poca relación
    señal-ruido que un modelo temporal solo detectará desviaciones extremas
    (riesgo R13 de la propuesta).
    """
    g = df.groupby("account")

    profile = pd.DataFrame({
        "periods": g.size(),
        "periods_with_movement": g["no_movement"].apply(lambda s: (s == 0).sum()),
        "mean_movement": g["net_movement"].mean(),
        "mean_abs_movement": g["net_movement"].apply(lambda s: s.abs().mean()),
        "std_movement": g["net_movement"].std(),
        "min_movement": g["net_movement"].min(),
        "max_movement": g["net_movement"].max(),
        "total_lines": g["line_count"].sum(),
        "first_period": g["period_date"].min(),
        "last_period": g["period_date"].max(),
    })

    profile["zero_share"] = 1 - profile["periods_with_movement"] / profile["periods"]
    profile["noise_ratio_all"] = (
        profile["std_movement"] / profile["mean_abs_movement"].replace(0, pd.NA)
    )

    # Ratio sobre los períodos CON movimiento. Es la medida correcta de
    # relación señal-ruido: separa "cuenta ruidosa" de "cuenta esporádica",
    # que exigen tratamientos distintos.
    active = df[df["no_movement"] == 0].groupby("account")["net_movement"]
    profile["std_active"] = active.std()
    profile["mean_abs_active"] = active.apply(lambda s: s.abs().mean())
    profile["noise_ratio"] = (
        profile["std_active"] / profile["mean_abs_active"].replace(0, pd.NA)
    )
    # Aproximación teórica del límite de detección
    profile["min_detectable"] = 2 * profile["std_movement"]

    return profile.sort_values("periods", ascending=False)


def to_matrix(df: pd.DataFrame, accounts=None) -> pd.DataFrame:
    """
    Formato ancho: filas = período, columnas = cuenta. Es lo que consumen los
    modelos temporales.

    La rejilla ya viene completa desde la vista (los meses sin movimiento son
    ceros informativos), así que los NaN que aparezcan corresponden a series
    que empiezan más tarde, no a huecos internos.
    """
    if accounts is not None:
        df = df[df["account"].isin(list(accounts))]

    matrix = (df.pivot_table(index="period_date", columns="account",
                             values="net_movement")
                .sort_index())

    missing = int(matrix.isna().sum().sum())
    if missing:
        print(f"Aviso: {missing:,} celdas vacías — series con inicio posterior "
              f"al del histórico. Decidir si recortar o imputar.")
    return matrix


def summarize(profile: pd.DataFrame, config: LoadConfig | None = None) -> None:
    """Imprime los números que validan el diseño del modelo."""
    config = config or LoadConfig()

    print(f"\nCuentas: {len(profile)}")
    for lo, hi, label in [(24, 999, "24+  (LSTM viable)"),
                          (12, 23, "12-23 (AE simple)"),
                          (0, 11, "<12  (solo baseline)")]:
        n = int(profile["periods"].between(lo, hi).sum())
        print(f"  {label:22s}: {n:4d}  ({100 * n / len(profile):5.1f}%)")

    print("\nRelación ruido/señal sobre períodos CON movimiento (R13):")
    nr = profile["noise_ratio"].dropna()
    for lo, hi, label in [(0, 0.5, "< 0.5  estable"),
                          (0.5, 1.0, "0.5-1  manejable"),
                          (1.0, 2.0, "1-2    ruidosa"),
                          (2.0, 1e9, "> 2    muy ruidosa")]:
        n = int(nr.between(lo, hi, inclusive="left").sum())
        print(f"  {label:20s}: {n:4d}  ({100 * n / len(nr):5.1f}%)")
 
    print("\nDispersión (períodos sin movimiento):")
    for lo, hi, label in [(0, 0.25, "< 25%   densa"),
                          (0.25, 0.50, "25-50%  regular"),
                          (0.50, 0.75, "50-75%  esporádica"),
                          (0.75, 1.01, "> 75%   muy esporádica")]:
        n = int(profile["zero_share"].between(lo, hi, inclusive="left").sum())
        print(f"  {label:22s}: {n:4d}  ({100 * n / len(profile):5.1f}%)")
 
    # Series aptas para el modelo temporal: historia suficiente Y no dominadas
    # por ceros. Una serie mayoritariamente vacía no aporta dinámica temporal
    # aunque tenga muchos períodos.
    viable = profile[(profile["periods"] >= config.min_periods)
                     & (profile["zero_share"] <= 0.5)]
    print(f"\nSeries viables para el modelo temporal "
          f"(>= {config.min_periods} períodos y <= 50% en cero): {len(viable)}")