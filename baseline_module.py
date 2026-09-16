"""
baseline.py
-----------
Línea base estadística para detección de anomalías en balances mensuales.

Es el PISO de comparación obligatorio: ningún modelo complejo se adopta si no
supera de forma consistente a estos detectores. Se implementan tres, elegidos
según lo que reveló el EDA:

  1. RobustZScore  — para cuentas densas. Mediana y MAD en vez de media y
                     desviación estándar, porque estas últimas son arrastradas
                     por los mismos valores extremos que se busca detectar.
  2. EventDetector — para cuentas esporádicas. Con más del 50% de períodos en
                     cero, la pregunta relevante no es "¿el monto es el
                     esperado?" sino "¿hubo movimiento cuando no debía?".
  3. SeasonalZScore — variante que compara cada mes contra el mismo mes de
                     otros años. El EDA mostró diciembre 10,8x sobre abril:
                     sin ajuste estacional, todos los cierres saldrían
                     marcados como anómalos.

Uso:
    from baseline import run_baseline
    scores, summary = run_baseline(df, profile)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Escala el MAD para hacerlo comparable a una desviación estándar normal.
MAD_SCALE = 1.4826

# Piso de materialidad en USD. Sin él, una cuenta con MAD cercano a cero
# produce puntajes astronómicos ante desviaciones triviales: en la primera
# corrida, un movimiento de 1.368 USD obtuvo puntaje 184.522 y desplazó del
# primer lugar a uno de 385 millones. El piso impide que la estabilidad
# estadística se confunda con relevancia contable.
#
# Debe ajustarse con el criterio del equipo de finanzas: es el monto por
# debajo del cual una desviación no amerita investigación (Bloque E del
# requerimiento de información).
MATERIALITY_USD = 1000.0


# =============================================================================
# Utilidades
# =============================================================================

def robust_stats(values: pd.Series,
                 materiality: float = MATERIALITY_USD) -> tuple[float, float]:
    """
    Devuelve (mediana, MAD escalado con piso de materialidad).

    El piso convierte el puntaje en "cuántas veces la materialidad" cuando la
    cuenta es muy estable, en lugar de dividir por un valor casi nulo.
    """
    median = values.median()
    mad = (values - median).abs().median() * MAD_SCALE
    return median, max(mad, materiality)


def classify_accounts(profile: pd.DataFrame,
                      zero_threshold: float = 0.5) -> pd.Series:
    """
    Separa las dos poblaciones que el EDA reveló como bimodales.

    La distribución de períodos sin movimiento no es un continuo: ~168 cuentas
    se mueven todos los meses y ~130 se mueven pocas veces al año, con muy poco
    en medio. Un solo modelo tendría que aprender que el cero es normal para
    unas y alarmante para otras.
    """
    return pd.Series(
        np.where(profile["zero_share"] <= zero_threshold, "dense", "sparse"),
        index=profile.index,
        name="account_type",
    )


# =============================================================================
# 1. Z-score robusto (cuentas densas)
# =============================================================================

def robust_zscore(df: pd.DataFrame, min_periods: int = 12,
                  materiality: float = MATERIALITY_USD) -> pd.DataFrame:
    """
    Puntúa cada período contra la propia historia de su cuenta.

    Se calcula sobre períodos con movimiento: incluir los ceros desplazaría la
    mediana hacia cero en las cuentas esporádicas y volvería anómalo cualquier
    movimiento.
    """
    out = []

    for account, g in df.groupby("account"):
        active = g[g["no_movement"] == 0]
        if len(active) < min_periods:
            continue

        median, mad = robust_stats(active["net_movement"], materiality)

        scored = g.copy()
        scored["median_ref"] = median
        scored["mad_ref"] = mad
        scored["score_z"] = (
            (scored["net_movement"] - median) / mad if mad > 0 else np.nan
        )
        out.append(scored)

    if not out:
        return pd.DataFrame()

    result = pd.concat(out, ignore_index=True)
    result["abs_score_z"] = result["score_z"].abs()
    return result


# =============================================================================
# 2. Z-score estacional
# =============================================================================

def seasonal_zscore(df: pd.DataFrame, min_years: int = 2,
                    materiality: float = MATERIALITY_USD) -> pd.DataFrame:
    """
    Compara cada período contra el MISMO MES de otros años.

    Corrige el efecto que el EDA cuantificó: diciembre mueve 10,8 veces más que
    abril. Sin este ajuste, cada cierre de ejercicio se marcaría como anomalía.

    Limitación: con ~3,5 años de historia hay solo 3 o 4 observaciones por mes
    calendario, así que la referencia estacional es poco precisa. Se reporta
    `n_same_month` para que el consumidor pondere la confianza.
    """
    d = df.copy()
    d["month"] = d["period_date"].dt.month
    out = []

    for (account, month), g in d.groupby(["account", "month"]):
        active = g[g["no_movement"] == 0]
        if len(active) < min_years:
            continue

        median, mad = robust_stats(active["net_movement"], materiality)

        scored = g.copy()
        scored["seasonal_median"] = median
        scored["seasonal_mad"] = mad
        scored["n_same_month"] = len(active)
        scored["score_seasonal"] = (
            (scored["net_movement"] - median) / mad if mad > 0 else np.nan
        )
        out.append(scored)

    if not out:
        return pd.DataFrame()

    result = pd.concat(out, ignore_index=True)
    result["abs_score_seasonal"] = result["score_seasonal"].abs()
    return result


# =============================================================================
# 3. Detector de eventos (cuentas esporádicas)
# =============================================================================

def event_detector(df: pd.DataFrame, sparse_accounts: list[str]) -> pd.DataFrame:
    """
    Para cuentas que se mueven pocas veces al año, lo informativo es la
    OCURRENCIA, no la magnitud.

    Dos señales:
      - `score_unexpected`: hubo movimiento en una cuenta que casi nunca se
        mueve. Cuanto más rara la actividad, mayor el puntaje.
      - `score_silence`  : la cuenta lleva sin moverse mucho más de lo habitual
        entre movimientos. Detecta interfaces caídas o cargas omitidas.
    """
    d = df[df["account"].isin(sparse_accounts)].copy()
    out = []

    for account, g in d.groupby("account"):
        g = g.sort_values("period_date").copy()

        activity_rate = (g["no_movement"] == 0).mean()
        if activity_rate == 0:
            continue

        # Rareza de que haya movimiento: -log de la tasa de actividad.
        g["score_unexpected"] = np.where(
            g["no_movement"] == 0, -np.log(max(activity_rate, 1e-6)), 0.0
        )

        # Períodos consecutivos sin movimiento, contra el intervalo habitual.
        streak, streaks = 0, []
        for is_empty in g["no_movement"]:
            streak = streak + 1 if is_empty else 0
            streaks.append(streak)
        g["silence_streak"] = streaks

        gaps = [s for s in streaks if s > 0]
        typical_gap = np.median(gaps) if gaps else 1
        g["score_silence"] = g["silence_streak"] / max(typical_gap, 1)

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# =============================================================================
# Orquestación
# =============================================================================

def run_baseline(df: pd.DataFrame,
                 profile: pd.DataFrame,
                 zero_threshold: float = 0.5,
                 top_k: int = 50,
                 materiality: float = MATERIALITY_USD) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta la línea base completa y devuelve (puntajes, resumen).

    `top_k` refleja la restricción operativa real: el equipo revisa un número
    limitado de alertas por ciclo. La métrica que importa es qué proporción de
    esas K son problemas reales, no la exactitud global.
    """
    types = classify_accounts(profile, zero_threshold)
    dense = types[types == "dense"].index.tolist()
    sparse = types[types == "sparse"].index.tolist()

    df_dense = df[df["account"].isin(dense)]

    z = robust_zscore(df_dense, materiality=materiality)
    seasonal = seasonal_zscore(df_dense, materiality=materiality)
    events = event_detector(df, sparse)

    # --- Consolidar en una tabla de puntajes ---
    keys = ["account", "period_name", "period_date", "net_movement"]
    scores = df[keys].copy()

    for source, cols in [(z, ["score_z", "abs_score_z", "median_ref"]),
                         (seasonal, ["score_seasonal", "abs_score_seasonal",
                                     "n_same_month"]),
                         (events, ["score_unexpected", "score_silence",
                                   "silence_streak"])]:
        if not source.empty:
            scores = scores.merge(
                source[["account", "period_name"] + cols],
                on=["account", "period_name"], how="left",
            )

    scores["account_type"] = scores["account"].map(types)

    # --- Desviación en USD: la magnitud contable de la anomalía -------------
    ref = scores.get("seasonal_median", pd.Series(np.nan, index=scores.index))
    ref = ref.fillna(scores.get("median_ref", pd.Series(np.nan, index=scores.index)))
    scores["deviation_usd"] = (scores["net_movement"] - ref.fillna(0)).abs()

    # --- Puntaje estadístico por tipo de cuenta -----------------------------
    stat = np.where(
        scores["account_type"] == "sparse",
        scores.get("score_unexpected", pd.Series(0.0, index=scores.index)).fillna(0)
        + scores.get("score_silence", pd.Series(0.0, index=scores.index)).fillna(0),
        scores.get("abs_score_seasonal", pd.Series(np.nan, index=scores.index))
              .fillna(scores.get("abs_score_z", pd.Series(np.nan, index=scores.index))),
    )
    scores["score_stat"] = stat

    # --- Puntaje operativo: rareza ponderada por materialidad ---------------
    # Un puntaje puramente estadístico ordena por rareza y deja arriba
    # desviaciones triviales en cuentas muy estables. Multiplicar por el
    # logaritmo de la desviación en USD incorpora la relevancia contable sin
    # que las cuentas grandes copen el ranking por su sola escala.
    scores["score"] = (
        scores["score_stat"].fillna(0)
        * np.log1p(scores["deviation_usd"].fillna(0) / materiality)
    )

    # --- Ranking dentro de cada tipo ---------------------------------------
    # Los puntajes de cuentas densas y esporádicas viven en escalas distintas;
    # mezclarlos hace que un tipo domine el ranking sin que sus casos sean más
    # graves. En la primera corrida, las 50 primeras alertas eran todas densas.
    scores["rank_within_type"] = (
        scores.groupby("account_type")["score"]
              .rank(ascending=False, method="first")
    )

    summary = {
        "dense_accounts": len(dense),
        "sparse_accounts": len(sparse),
        "scored_rows": int(scores["score"].notna().sum()),
        "flagged_z_gt_3": int((scores.get("abs_score_z", pd.Series(dtype=float)) > 3).sum()),
        "flagged_seasonal_gt_3": int(
            (scores.get("abs_score_seasonal", pd.Series(dtype=float)) > 3).sum()),
        "top_k": top_k,
        "materiality_usd": materiality,
        "median_deviation_top_k": float(
            scores.nlargest(top_k, "score")["deviation_usd"].median()),
    }

    return scores.sort_values("score", ascending=False), summary


def balanced_top(scores: pd.DataFrame, k: int = 50,
                 sparse_share: float = 0.3) -> pd.DataFrame:
    """
    Top-K con representación garantizada de ambos tipos de cuenta.

    Sin esto, las cuentas esporádicas nunca alcanzan el ranking global aunque
    presenten comportamientos claramente anómalos, porque sus puntajes están
    en otra escala.
    """
    n_sparse = int(k * sparse_share)
    n_dense = k - n_sparse

    dense = scores[scores["account_type"] == "dense"].nlargest(n_dense, "score")
    sparse = scores[scores["account_type"] == "sparse"].nlargest(n_sparse, "score")

    return pd.concat([dense, sparse]).sort_values("score", ascending=False)


def detection_limits(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Anomalía mínima detectable por serie, en USD.

    Traduce el umbral estadístico a una magnitud contable concreta: cuán grande
    debe ser un error para que el detector lo vea. Es lo que permite decirle al
    negocio qué cubre el sistema y qué no.
    """
    limits = profile[["periods", "mean_abs_active", "std_active",
                      "noise_ratio", "zero_share"]].copy()
    limits["min_detectable_z3"] = 3 * limits["std_active"]
    limits["ratio_to_typical"] = (
        limits["min_detectable_z3"] / limits["mean_abs_active"].replace(0, np.nan)
    )
    return limits.sort_values("ratio_to_typical", ascending=False)


def diverse_top(scores: pd.DataFrame, k: int = 50,
                max_per_account: int = 2,
                sparse_share: float = 0.3) -> pd.DataFrame:
    """
    Top-K con límite de alertas por cuenta y representación de ambos tipos.

    Sin el límite, unas pocas cuentas volátiles copan el ranking: en la primera
    corrida, 22 alertas correspondían a solo 15 cuentas, con tres apariciones
    de una misma. Las alertas repetidas de una cuenta suelen compartir
    explicación, de modo que el analista revisa menos casos distintos de los
    que sugiere el conteo.
    """
    n_sparse = int(k * sparse_share)
    n_dense = k - n_sparse

    def pick(subset: pd.DataFrame, n: int) -> pd.DataFrame:
        ranked = (subset.sort_values("score", ascending=False)
                        .groupby("account", sort=False)
                        .head(max_per_account))
        return ranked.nlargest(n, "score")

    dense = pick(scores[scores["account_type"] == "dense"], n_dense)
    sparse = pick(scores[scores["account_type"] == "sparse"], n_sparse)

    return pd.concat([dense, sparse]).sort_values("score", ascending=False)


def find_reversals(df: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """
    Pares de reverso: un movimiento seguido de otro casi idéntico y de signo
    contrario en el período siguiente.

    Es una firma mucho más específica que un valor atípico aislado, y
    corresponde al tipo "acumulación no reversada" de la taxonomía cuando el
    reverso ocurre tarde o no ocurre. Detectados de forma directa, sin modelo.
    """
    d = df.sort_values(["account", "period_date"]).copy()
    d["prev_movement"] = d.groupby("account")["net_movement"].shift(1)
    d["prev_period"] = d.groupby("account")["period_name"].shift(1)

    residual = (d["net_movement"] + d["prev_movement"]).abs()
    magnitude = d["net_movement"].abs()

    d["is_reversal"] = (
        (d["net_movement"] * d["prev_movement"] < 0)
        & (residual < tolerance * magnitude)
        & (magnitude > MATERIALITY_USD)
    )

    cols = ["account", "prev_period", "period_name",
            "prev_movement", "net_movement"]
    out = d.loc[d["is_reversal"], cols].copy()
    out["amount"] = out["net_movement"].abs()
    return out.sort_values("amount", ascending=False)


def classify_reversals(reversals: pd.DataFrame,
                       routine_threshold: int = 2) -> pd.DataFrame:
    """
    Separa los reversos rutinarios de los que rompen el patrón.

    Un reverso no es anómalo por sí mismo: revertir una acumulación de cierre
    en enero es práctica contable estándar. Lo informativo es el reverso que
    NO encaja en el patrón habitual de su cuenta.

    Dos criterios de rutina:
      - Cierre de ejercicio: el par diciembre -> enero.
      - Recurrencia: la cuenta acumula al menos `routine_threshold` reversos
        en el histórico, de modo que revertir forma parte de su operación
        normal aunque no sea en enero.
    """
    out = reversals.copy()

    month_prev = out["prev_period"].str.extract(r"^([A-Za-z]{3})", expand=False).str.upper()
    month_curr = out["period_name"].str.extract(r"^([A-Za-z]{3})", expand=False).str.upper()
    out["is_year_end"] = (month_prev == "DEC") & (month_curr == "JAN")

    counts = out.groupby("account").size()
    out["account_reversals"] = out["account"].map(counts)
    out["is_frequent"] = out["account_reversals"] >= routine_threshold

    out["category"] = np.where(
        out["is_year_end"], "rutina_cierre",
        np.where(out["is_frequent"], "rutina_cuenta", "atipico"),
    )
    return out.sort_values(["category", "amount"], ascending=[True, False])