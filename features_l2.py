"""
features_l2.py
==============

Construcción de la matriz de features para el autoencoder de Nivel 2
(análisis temporal de balances mensuales).

Diseño
------
El autoencoder reconstruye un vector de features por cada (serie, período).
La lógica de dominio vive aquí, no en el modelo:

  1. Transformación logaritmica: sign(x) * log1p(|x|). Los montos abarcan
     seis órdenes de magnitud y pueden ser negativos, así que ni log ni
     log1p simples sirven.
  2. Residuo estacional: a cada punto se le resta el centro robusto (mediana)
     de su mismo mes-de-año dentro de la serie, escalado por la MAD con un
     piso de materialidad. Es la misma idea del seasonal z-score del baseline;
     así el autoencoder no gasta su escasa capacidad aprendiendo que diciembre
     es ~10x abril.
  3. Contexto temporal: lags cortos (t-1..t-3) y estadísticos móviles.
  4. Codificación cíclica del mes (sin/cos) por si la desestacionalización
     queda imperfecta.

Flujo sin fuga temporal (featurizar-luego-cortar)
-------------------------------------------------
    cfg = FeatureConfig(value_col="net_movement")     # ajusta a tu vista
    fb  = FeatureBuilder(cfg)

    # 1. Selección de series densas sobre la historia COMPLETA
    dense = FeatureBuilder.select_dense_series(full_df, cfg)

    # 2. Parámetros aprendidos SOLO con train
    fb.fit(full_df[full_df.period_date < split], selected_series=dense)

    # 3. Featurizar la serie completa -> lags continuos en la frontera
    X_all = fb.transform(full_df)

    # 4. Cortar la matriz de features por fecha
    X_train = X_all[X_all.period_date <  split]
    X_test  = X_all[X_all.period_date >= split]

    cols = fb.feature_cols   # columnas listas para el autoencoder

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuración
# =============================================================================
@dataclass
class FeatureConfig:
    """Mapeo de columnas y parámetros del constructor de features.

    """

    # --- Identidad de la serie y eje temporal ---
    # La vista viene consolidada a ou_group y load_series filtra a un solo
    # ou_group ('UNV'), así que la serie se identifica por 'account'.
    series_keys: list[str] = field(default_factory=lambda: ["account"])
    period_col: str = "period_name"          # etiqueta del período (p.ej. 'May-24')
    period_date_col: str = "period_date"     # fecha para ordenar cronológicamente

    # --- Señal principal ---
    value_col: str = "net_movement"          # movimiento neto del período (USD funcional)

    # --- Señales opcionales (se ignoran si la columna no existe) ---
    level_col: Optional[str] = "running_balance"    # nivel / balance acumulado
    volume_col: Optional[str] = "line_count"        # volumen de actividad
    source_col: Optional[str] = "active_ou_count"   # diversidad de OUs activas

    # --- Bandera de período de ajuste ---
    is_adjustment_col: Optional[str] = None

    # --- Parámetros de modelado ---
    min_periods: int = 24        # solo series con historial suficiente (densas)
    max_lag: int = 3             # lags cortos: se pierden los primeros max_lag puntos
    roll_window: int = 3         # ventana de estadísticos móviles
    materiality_floor: float = 1_000.0   # piso para la MAD estacional (USD), como en baseline
    channel_floor: float = 0.5   # piso (en unidades slog) para la MAD por-cuenta de los
                                 # canales extra (line_count, running_balance, ...)


# =============================================================================
# Utilidades numéricas
# =============================================================================
def signed_log1p(x: pd.Series) -> pd.Series:
    """log-firmado: preserva el signo y comprime seis órdenes de magnitud."""
    return np.sign(x) * np.log1p(np.abs(x))


def _month_of_year(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.month


def _mad(values: np.ndarray) -> float:
    """Desviación absoluta mediana, escalada a equivalente-sigma (x1.4826)."""
    med = np.nanmedian(values)
    return 1.4826 * np.nanmedian(np.abs(values - med))


# =============================================================================
# Constructor de features
# =============================================================================
class FeatureBuilder:
    """Construye la matriz de features con separación fit/transform."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.cfg = config or FeatureConfig()
        self.feature_cols: list[str] = []
        # Parámetros aprendidos en fit:
        self._series_selected: set[tuple] = set()   # series densas escogidas
        self._seasonal_center: dict = {}            # (serie, mes) -> mediana de slog
        self._seasonal_scale: dict = {}             # (serie, mes) -> MAD de slog (con piso)
        self._chan_center: dict = {}                # (serie, col) -> mediana de slog(col)
        self._chan_scale: dict = {}                 # (serie, col) -> MAD de slog(col) (con piso)
        self._feat_mean: pd.Series | None = None
        self._feat_std: pd.Series | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Validación
    # ------------------------------------------------------------------ #
    def _validate(self, df: pd.DataFrame) -> None:
        cfg = self.cfg
        required = [*cfg.series_keys, cfg.period_col, cfg.period_date_col, cfg.value_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                "Faltan columnas requeridas en el DataFrame: "
                f"{missing}. Ajusta FeatureConfig para que coincida con tu vista. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    def _optional_present(self, df: pd.DataFrame) -> list[str]:
        cfg = self.cfg
        opts = [cfg.level_col, cfg.volume_col, cfg.source_col]
        return [c for c in opts if c and c in df.columns]

    # ------------------------------------------------------------------ #
    # Preparación común: exclusión de ajustes + orden cronológico
    # (NO filtra por longitud; la selección de series es un paso aparte)
    # ------------------------------------------------------------------ #
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()
        if cfg.is_adjustment_col and cfg.is_adjustment_col in out.columns:
            out = out[out[cfg.is_adjustment_col] != 1]
        out[cfg.period_date_col] = pd.to_datetime(out[cfg.period_date_col])
        out = out.sort_values([*cfg.series_keys, cfg.period_date_col]).reset_index(drop=True)
        return out

    def _key_tuples(self, df: pd.DataFrame) -> list[tuple]:
        return list(map(tuple, df[self.cfg.series_keys].to_numpy().tolist()))

    def _restrict(self, df: pd.DataFrame, keyset: set[tuple]) -> pd.DataFrame:
        """Filtra las filas a las series presentes en keyset."""
        if not keyset:
            return df.iloc[0:0].copy()
        mask = np.array([t in keyset for t in self._key_tuples(df)])
        return df[mask].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Selección de series densas (sobre la historia completa)
    # ------------------------------------------------------------------ #
    @staticmethod
    def select_dense_series(df: pd.DataFrame, config: FeatureConfig) -> set[tuple]:
        """Devuelve el conjunto de claves de serie con >= min_periods períodos,
        excluyendo períodos de ajuste. Aplícalo sobre la historia COMPLETA."""
        out = df.copy()
        if config.is_adjustment_col and config.is_adjustment_col in out.columns:
            out = out[out[config.is_adjustment_col] != 1]
        counts = out.groupby(config.series_keys).size().reset_index(name="_n")
        dense = counts[counts["_n"] >= config.min_periods][config.series_keys]
        return set(map(tuple, dense.to_numpy().tolist()))

    # ------------------------------------------------------------------ #
    # fit: aprende centros estacionales y estandarización (solo con train)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        df: pd.DataFrame,
        selected_series: Optional[Iterable[tuple]] = None,
    ) -> "FeatureBuilder":
        self._validate(df)
        cfg = self.cfg

        # Selección de series: explícita (recomendado, sobre historia completa)
        # o inferida del df de fit como conveniencia.
        if selected_series is None:
            selected_series = self.select_dense_series(df, cfg)
        self._series_selected = set(map(tuple, selected_series))

        prep = self._restrict(self._prepare(df), self._series_selected)
        if prep.empty:
            raise ValueError(
                "El conjunto de fit quedó vacío tras restringir a las series densas. "
                "Revisa min_periods y que 'selected_series' corresponda a estos datos."
            )

        # Centros y escalas estacionales por (serie, mes-de-año), sobre train
        slog = signed_log1p(prep[cfg.value_col])
        month = _month_of_year(prep[cfg.period_date_col])
        tmp = prep[cfg.series_keys].copy()
        tmp["_slog"] = slog.values
        tmp["_month"] = month.values
        grp = tmp.groupby([*cfg.series_keys, "_month"])["_slog"]
        self._seasonal_center = grp.median().to_dict()
        self._seasonal_scale = grp.apply(lambda s: _mad(s.values)).to_dict()

        # Centros y escalas POR CUENTA de cada canal extra (line_count, etc.),
        # para volverlos relativos a la cuenta en vez de absolutos globales.
        self._chan_center, self._chan_scale = {}, {}
        for col in self._optional_present(prep):
            cslog = signed_log1p(prep[col])
            t = prep[cfg.series_keys].copy()
            t["_cs"] = cslog.values
            cg = t.groupby(cfg.series_keys)["_cs"]
            for key, med in cg.median().to_dict().items():
                key_t = key if isinstance(key, tuple) else (key,)
                self._chan_center[(key_t, col)] = med
            for key, mad in cg.apply(lambda s: _mad(s.values)).to_dict().items():
                key_t = key if isinstance(key, tuple) else (key,)
                self._chan_scale[(key_t, col)] = mad

        # Estandarización: media/desv de las features sobre train
        feats = self._engineer(prep, fit_mode=True)
        self._feat_mean = feats[self.feature_cols].mean()
        self._feat_std = feats[self.feature_cols].std().replace(0, 1.0)

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # transform: aplica parámetros aprendidos (usar sobre la serie completa)
    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Llama a fit() antes de transform().")
        self._validate(df)
        prep = self._restrict(self._prepare(df), self._series_selected)
        feats = self._engineer(prep, fit_mode=False)
        feats[self.feature_cols] = (
            feats[self.feature_cols] - self._feat_mean
        ) / self._feat_std
        return feats

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conveniencia para el caso sin split (fit y transform sobre el mismo df)."""
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------ #
    # Ingeniería de features (compartida por fit y transform)
    # ------------------------------------------------------------------ #
    def _engineer(self, prep: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        cfg = self.cfg
        df = prep.copy()

        df["slog"] = signed_log1p(df[cfg.value_col])
        df["_month"] = _month_of_year(df[cfg.period_date_col])

        # Residuo estacional con centros/escalas aprendidos
        keys = list(zip(*[df[k] for k in cfg.series_keys], df["_month"]))
        centers = np.array([self._seasonal_center.get(k, np.nan) for k in keys])
        scales = np.array([self._seasonal_scale.get(k, np.nan) for k in keys])
        floor_log = np.log1p(cfg.materiality_floor)   # piso en escala log
        scales = np.where(np.isnan(scales) | (scales < floor_log), floor_log, scales)
        centers = np.where(np.isnan(centers), df["slog"].values, centers)
        df["resid_seasonal"] = (df["slog"].values - centers) / scales

        feature_cols: list[str] = ["slog", "resid_seasonal"]

        # Codificación cíclica del mes
        df["month_sin"] = np.sin(2 * np.pi * df["_month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["_month"] / 12)
        feature_cols += ["month_sin", "month_cos"]

        # Lags y estadísticos móviles del residuo, por serie
        g = df.groupby(cfg.series_keys)["resid_seasonal"]
        for lag in range(1, cfg.max_lag + 1):
            col = f"resid_lag{lag}"
            df[col] = g.shift(lag)
            feature_cols.append(col)
        df["resid_roll_mean"] = g.transform(
            lambda s: s.rolling(cfg.roll_window, min_periods=1).mean()
        )
        df["resid_roll_std"] = g.transform(
            lambda s: s.rolling(cfg.roll_window, min_periods=2).std()
        ).fillna(0.0)
        feature_cols += ["resid_roll_mean", "resid_roll_std"]

        # Señales opcionales, RELATIVAS A LA CUENTA (nivel, volumen, origen).
        # slog(col) menos la mediana por-cuenta, escalado por su MAD con piso.
        # Así una anomalía relativa a la cuenta (p.ej. 40 -> 320 líneas) resalta,
        # aunque en términos globales 320 líneas sea normal para otras cuentas.
        floor_chan = cfg.channel_floor
        for col in self._optional_present(df):
            cslog = signed_log1p(df[col]).values
            keys_c = list(zip(*[df[k] for k in cfg.series_keys]))
            if len(cfg.series_keys) == 1:
                keys_c = [(k,) for k in df[cfg.series_keys[0]]]
            centers = np.array([self._chan_center.get((k, col), np.nan) for k in keys_c])
            scales = np.array([self._chan_scale.get((k, col), np.nan) for k in keys_c])
            scales = np.where(np.isnan(scales) | (scales < floor_chan), floor_chan, scales)
            centers = np.where(np.isnan(centers), cslog, centers)
            fname = f"resid_{col}"
            df[fname] = (cslog - centers) / scales
            feature_cols.append(fname)

        # Descartar los primeros max_lag puntos de cada serie (lags NaN)
        df = df.dropna(subset=[f"resid_lag{cfg.max_lag}"]).reset_index(drop=True)

        if fit_mode:
            self.feature_cols = feature_cols

        keep = [
            *cfg.series_keys,
            cfg.period_col,
            cfg.period_date_col,
            cfg.value_col,
            *feature_cols,
        ]
        return df[keep].copy()


# =============================================================================
# Ejemplo de uso / sanity check
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    periods = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for acct in ["10010", "14024", "76125"]:
        season = 5_000 * (1 + np.sin(2 * np.pi * periods.month / 12))
        noise = rng.normal(0, 800, len(periods))
        vals = season + noise
        running = 0.0
        for d, v in zip(periods, vals):
            running += float(v)
            rows.append(
                dict(
                    account=acct,
                    ou_group="UNV",
                    period_name=d.strftime("%b-%y"),
                    period_date=d,
                    net_movement=float(v),
                    line_count=int(rng.integers(5, 50)),
                    active_ou_count=int(rng.integers(1, 6)),
                    no_movement=0,
                    running_balance=running,
                )
            )
    demo = pd.DataFrame(rows)

    cfg = FeatureConfig()   # defaults ya apuntan a las columnas reales de la vista
    fb = FeatureBuilder(cfg)
    split = pd.Timestamp("2024-07-01")

    # Flujo featurizar-luego-cortar
    dense = FeatureBuilder.select_dense_series(demo, cfg)
    fb.fit(demo[demo.period_date < split], selected_series=dense)
    X_all = fb.transform(demo)
    X_train = X_all[X_all.period_date < split]
    X_test = X_all[X_all.period_date >= split]

    print("Series densas seleccionadas:", len(dense))
    print("Columnas de features:", fb.feature_cols)
    print("Shape train:", X_train.shape, "| test:", X_test.shape)
    print(X_train[fb.feature_cols].describe().T[["mean", "std", "min", "max"]].round(2))