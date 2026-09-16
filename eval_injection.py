"""
eval_injection.py
=================

Evaluación por inyección sintética para detección de anomalías de Nivel 2.

Problema
--------
Casi no hay etiquetas (un candidato real, la 14024). Sin verdad-terreno no se
puede medir cuantitativamente qué detecta cada modelo. La solución estándar y
defendible: tomar las series limpias, **inyectar anomalías conocidas** de tipos
concretos, y medir qué fracción recupera cada detector y a qué costo de falsos
positivos.

Diseño
------
El arnés es **agnóstico al detector**. Recibe un `scorer`: una función que toma
un DataFrame crudo (con net_movement, etc.) y devuelve un score por
(cuenta, período). Así el AE y el baseline se evalúan sobre la MISMA inyección.

  - Para el AE:      make_ae_scorer(feature_builder, ae, ...)
  - Para el baseline: escribes un wrapper equivalente sobre tu baseline.py

Tipos de anomalía (mapeados a modos de error contable reales)
-------------------------------------------------------------
  spike       : movimiento único de gran magnitud (asiento erróneo/duplicado)
  reversal    : +m en t y -m en t+1 (asiento reversado; el caso 14024)
  level_shift : desplazamiento sostenido durante una ventana (reclasificación)
  dropout     : período con movimiento forzado a cero (omisión de asiento)
  sign_flip   : signo invertido (débito/crédito intercambiados)

La magnitud se calibra en **múltiplos de la escala robusta de cada serie**
(MAD con piso de materialidad), no en dólares absolutos. Así la misma "fuerza"
de anomalía es comparable entre cuentas grandes y pequeñas, y barriendo la
magnitud se obtiene la curva de detección vs. tamaño — el piso de detección
del riesgo R13.

Detalle crítico
---------------
Tras inyectar se **re-featuriza con el MISMO FeatureBuilder ya ajustado** (no se
reajusta): la inyección no debe filtrarse a los centros estacionales ni a la
estandarización. Por eso importaba la separación fit/transform.

Convención: documentación en español, código e identificadores en inglés.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

# Un scorer devuelve un DataFrame con [*series_keys, period_col, "score"].
Scorer = Callable[[pd.DataFrame], pd.DataFrame]


# =============================================================================
# Configuración
# =============================================================================
@dataclass
class InjectionConfig:
    types: tuple[str, ...] = ("spike", "reversal", "level_shift", "dropout", "sign_flip")
    magnitude: float = 8.0            # múltiplos de la escala robusta de la serie
    n_injections: int = 60            # celdas inyectadas por tipo y repetición
    shift_len: int = 3                # duración de level_shift (períodos)
    n_repeats: int = 5
    budget: float = 0.05              # presupuesto de alerta (top 5%) para recall@budget
                                      # OJO: si la densidad de inyección supera el budget,
                                      # el recall queda topado. Prioriza AUC/AP.
    min_history: int = 4              # posición mínima en la serie (>= max_lag+1 del builder)
    materiality_floor: float = 1_000.0
    seed: int = 42


# =============================================================================
# Métricas (numpy puro, sin sklearn)
# =============================================================================
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Rangos promediando empates (equivalente a scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1  # rango promedio (1-indexado)
        i = j + 1
    return ranks


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC vía Mann-Whitney U. 0.5 = azar, 1.0 = separación perfecta."""
    y = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_ranks(np.asarray(scores, dtype=float))
    return (r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (área bajo precision-recall). Más informativa que ROC
    bajo fuerte desbalance de clases, que es nuestro caso (pocas inyecciones)."""
    y = np.asarray(y_true).astype(int)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def recall_at_budget(y_true: np.ndarray, scores: np.ndarray, budget: float) -> float:
    """Fracción de inyecciones recuperadas si se alerta el top-`budget` por score."""
    y = np.asarray(y_true).astype(bool)
    if y.sum() == 0:
        return float("nan")
    thr = np.quantile(scores, 1 - budget)
    flagged = np.asarray(scores) >= thr
    return float(flagged[y].mean())


# =============================================================================
# Escala robusta por serie (para calibrar la magnitud de inyección)
# =============================================================================
def robust_scale_per_series(
    df: pd.DataFrame, series_keys: Sequence[str], value_col: str, floor: float
) -> dict[tuple, float]:
    def _mad(s):
        med = np.median(s)
        return 1.4826 * np.median(np.abs(s - med))

    scales = df.groupby(list(series_keys))[value_col].apply(lambda s: _mad(s.values))
    out = {}
    for key, val in scales.items():
        key_t = key if isinstance(key, tuple) else (key,)
        out[key_t] = max(float(val), floor)
    return out


# =============================================================================
# Inyección
# =============================================================================
def inject(
    df: pd.DataFrame,
    inj_type: str,
    cfg: InjectionConfig,
    rng: np.random.Generator,
    scales: dict[tuple, float],
    series_keys: Sequence[str],
    period_col: str,
    period_date_col: str,
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inyecta `inj_type` sobre una copia. Devuelve (df_perturbado, etiquetas)."""
    out = df.copy()
    out = out.sort_values([*series_keys, period_date_col]).reset_index(drop=True)
    out["_pos"] = out.groupby(list(series_keys)).cumcount()
    out["_len"] = out.groupby(list(series_keys))[value_col].transform("size")

    # Mapa (clave_serie, posición) -> índice de fila
    keys_arr = list(map(tuple, out[list(series_keys)].to_numpy().tolist()))
    pos_index = {(k, p): i for i, (k, p) in enumerate(zip(keys_arr, out["_pos"].tolist()))}

    # Espacio necesario después de la posición según el tipo
    tail = {"reversal": 1, "level_shift": cfg.shift_len - 1}.get(inj_type, 0)
    eligible = out[(out["_pos"] >= cfg.min_history) & (out["_pos"] < out["_len"] - tail)]

    n = min(cfg.n_injections, len(eligible))
    chosen = rng.choice(eligible.index.to_numpy(), size=n, replace=False)

    labels = []
    for i in chosen:
        key = keys_arr[i]
        m = cfg.magnitude * scales[key]

        if inj_type == "spike":
            out.at[i, value_col] += rng.choice([-1.0, 1.0]) * m
            labels.append(i)
        elif inj_type == "sign_flip":
            out.at[i, value_col] = -out.at[i, value_col]
            labels.append(i)
        elif inj_type == "dropout":
            out.at[i, value_col] = 0.0
            labels.append(i)
        elif inj_type == "reversal":
            j = pos_index[(key, out.at[i, "_pos"] + 1)]
            out.at[i, value_col] += m
            out.at[j, value_col] -= m
            labels.extend([i, j])
        elif inj_type == "level_shift":
            for step in range(cfg.shift_len):
                j = pos_index[(key, out.at[i, "_pos"] + step)]
                out.at[j, value_col] += m
                labels.append(j)
        else:
            raise ValueError(f"Tipo de inyección desconocido: {inj_type}")

    label_df = out.loc[sorted(set(labels)), [*series_keys, period_col]].copy()
    label_df["injected"] = 1
    return out.drop(columns=["_pos", "_len"]), label_df


# =============================================================================
# Adaptador del AE al contrato de scorer
# =============================================================================
def make_ae_scorer(feature_builder, ae, series_keys: Sequence[str], period_col: str) -> Scorer:
    """Envuelve (FeatureBuilder ya ajustado + AnomalyAE entrenado) como scorer."""
    def scorer(df: pd.DataFrame) -> pd.DataFrame:
        X = feature_builder.transform(df)          # re-featuriza con params limpios
        err = ae.reconstruction_error(X)
        res = X[[*series_keys, period_col]].copy()
        res["score"] = err
        return res
    return scorer


# =============================================================================
# Evaluación
# =============================================================================
def evaluate(
    df_clean: pd.DataFrame,
    scorer: Scorer,
    cfg: InjectionConfig,
    series_keys: Sequence[str],
    period_col: str = "period_name",
    period_date_col: str = "period_date",
    value_col: str = "net_movement",
    detector_name: str = "detector",
) -> pd.DataFrame:
    """Corre inyección + scoring + métricas sobre n_repeats, por tipo de anomalía.

    Devuelve un DataFrame ordenado con ROC-AUC, AP y recall@budget
    (media ± desviación entre repeticiones) por tipo.
    """
    scales = robust_scale_per_series(df_clean, series_keys, value_col, cfg.materiality_floor)
    key_cols = list(series_keys)
    records = []

    for rep in range(cfg.n_repeats):
        rng = np.random.default_rng(cfg.seed + rep)
        for inj_type in cfg.types:
            perturbed, labels = inject(
                df_clean, inj_type, cfg, rng, scales,
                series_keys, period_col, period_date_col, value_col,
            )
            scored = scorer(perturbed)                      # [keys, period, score]
            merged = scored.merge(labels, on=[*key_cols, period_col], how="left")
            merged["injected"] = merged["injected"].fillna(0).astype(int)

            y = merged["injected"].to_numpy()
            s = merged["score"].to_numpy()
            records.append({
                "detector": detector_name,
                "inj_type": inj_type,
                "magnitude": cfg.magnitude,
                "n_pos": int(y.sum()),
                "n_scored": int(len(y)),
                "density": round(y.sum() / len(y), 4),   # tasa base (piso del AP)
                "roc_auc": roc_auc(y, s),
                "ap": average_precision(y, s),
                "recall_at_budget": recall_at_budget(y, s, cfg.budget),
            })

    raw = pd.DataFrame(records)
    agg = (raw.groupby("inj_type")
              .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                   ap_mean=("ap", "mean"), ap_std=("ap", "std"),
                   recall_mean=("recall_at_budget", "mean"),
                   recall_std=("recall_at_budget", "std"),
                   density=("density", "mean"),
                   n_pos=("n_pos", "mean"))
              .reset_index())
    agg.insert(0, "detector", detector_name)
    return agg.round(3)


def sweep_magnitude(
    df_clean: pd.DataFrame,
    scorer: Scorer,
    cfg: InjectionConfig,
    magnitudes: Sequence[float],
    series_keys: Sequence[str],
    **kwargs,
) -> pd.DataFrame:
    """Repite `evaluate` variando la magnitud: curva de detección vs. tamaño (R13)."""
    frames = []
    for mag in magnitudes:
        cfg_m = InjectionConfig(**{**cfg.__dict__, "magnitude": mag})
        res = evaluate(df_clean, scorer, cfg_m, series_keys, **kwargs)
        res["magnitude"] = mag
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Demo de punta a punta con un scorer numpy (sin torch)
# =============================================================================
if __name__ == "__main__":
    # Scorer de reemplazo: z robusto por serie sobre net_movement. Sirve para
    # probar el arnés sin torch; en producción se usa make_ae_scorer.
    def demo_robustz_scorer(df, series_keys=("account",), period_col="period_name",
                            value_col="net_movement", floor=1000.0):
        def _score(s):
            med = np.median(s)
            mad = max(1.4826 * np.median(np.abs(s - med)), floor)
            return np.abs(s - med) / mad
        d = df.copy()
        d["score"] = d.groupby(list(series_keys))[value_col].transform(
            lambda s: _score(s.values))
        return d[[*series_keys, period_col, "score"]]

    # Datos sintéticos limpios
    rng = np.random.default_rng(0)
    periods = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for acct in [f"acc_{k:02d}" for k in range(60)]:
        season = 5_000 * (1 + np.sin(2 * np.pi * periods.month / 12))
        vals = season + rng.normal(0, 800, len(periods))
        for d, v in zip(periods, vals):
            rows.append(dict(account=acct, period_name=d.strftime("%b-%y"),
                             period_date=d, net_movement=float(v)))
    clean = pd.DataFrame(rows)

    cfg = InjectionConfig(n_injections=40, n_repeats=3)
    results = evaluate(clean, demo_robustz_scorer, cfg, series_keys=["account"],
                       detector_name="robust_z(demo)")
    print("Detección por tipo de anomalía (magnitud = 8x escala robusta):\n")
    print(results.to_string(index=False))

    print("\nBarrido de magnitud (recall@budget por tipo):\n")
    sweep = sweep_magnitude(clean, demo_robustz_scorer, cfg,
                            magnitudes=[2, 4, 8, 16], series_keys=["account"])
    pivot = sweep.pivot(index="inj_type", columns="magnitude", values="recall_mean")
    print(pivot.round(2).to_string())
