"""
lstm_autoencoder.py
===================

LSTM-Autoencoder (estilo EncDec-AD) para detección de anomalías de Nivel 2.

Modelo temporal protagonista, complementario al autoencoder denso (que actúa
como control). A diferencia del AE denso —un vector por período—, el LSTM
consume ventanas 3D `[n_ventanas, W, canales]`: secuencias de W meses
consecutivos por cuenta.

Diseño (decidido en el análisis)
--------------------------------
  - W = 12 (un ciclo anual).
  - Canales = las mismas features del AE denso (multivariante), reutilizando
    signed-log, residuo estacional y canales relativos a la cuenta de
    features_l2. Misma información, arquitectura distinta -> comparación limpia.
  - El AE reconstruye la VENTANA COMPLETA (eso fuerza la compresión de la
    dinámica normal en el cuello de botella), pero el score de anomalía se lee
    en el ÚLTIMO período de la ventana: un score por (cuenta, período), sin
    promediar entre ventanas solapadas, y con la señal concentrada en el
    período objetivo. Configurable vía cfg.score_mode.

Cobertura
---------
Cada período se puntúa con la ventana que termina en él, así que se pierden los
primeros W-1 períodos de cada serie (con W=12, una serie de 24 deja 13 puntos).

Reutiliza el arnés de eval_injection sin cambios vía make_lstm_scorer.
El LSTM consume la matriz YA estandarizada de features_l2.FeatureBuilder.

Convención: documentación en español, código e identificadores en inglés.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


# =============================================================================
# Configuración
# =============================================================================
@dataclass
class LSTMConfig:
    window: int = 12                 # W: longitud de la ventana (un ciclo anual)
    hidden: int = 32                 # unidades ocultas del LSTM
    bottleneck: int = 8              # dimensión del latente
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 15               # early stopping sobre validación
    batch_size: int = 64
    val_fraction: float = 0.15
    score_mode: str = "last"         # "last" (último período) o "window" (toda la ventana)
    seed: int = 42


# =============================================================================
# Construcción de ventanas (sin torch: testeable de forma independiente)
# =============================================================================
def build_windows(
    X: pd.DataFrame,
    feature_cols: Sequence[str],
    series_keys: Sequence[str],
    period_date_col: str,
    W: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Convierte la matriz por período en ventanas 3D [n, W, C].

    Devuelve (windows, meta) donde meta tiene una fila por ventana con las
    columnas de X del ÚLTIMO período de esa ventana (para alinear el score).
    Las ventanas nunca cruzan el límite de una cuenta.
    """
    X = X.sort_values([*series_keys, period_date_col])
    windows, metas = [], []
    for _, g in X.groupby(list(series_keys), sort=False):
        arr = g[list(feature_cols)].to_numpy(dtype=np.float32)
        n = len(g)
        if n < W:
            continue
        for end in range(W - 1, n):
            windows.append(arr[end - W + 1: end + 1])   # [W, C]
            metas.append(g.iloc[end])                   # último período
    if not windows:
        empty = X.iloc[0:0].copy()
        return np.empty((0, W, len(feature_cols)), dtype=np.float32), empty
    return np.stack(windows), pd.DataFrame(metas).reset_index(drop=True)


# =============================================================================
# Red: EncDec-AD (encoder LSTM -> latente -> decoder LSTM -> reconstrucción)
# =============================================================================
class _LSTMAE(nn.Module):
    def __init__(self, n_channels: int, cfg: LSTMConfig):
        super().__init__()
        self.W = cfg.window
        self.enc = nn.LSTM(n_channels, cfg.hidden, batch_first=True)
        self.to_latent = nn.Linear(cfg.hidden, cfg.bottleneck)
        self.from_latent = nn.Linear(cfg.bottleneck, cfg.hidden)
        self.dec = nn.LSTM(cfg.bottleneck, cfg.hidden, batch_first=True)
        self.out = nn.Linear(cfg.hidden, n_channels)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, W, C]
        _, (h, _) = self.enc(x)
        z = self.to_latent(self.drop(h[-1]))              # [B, bottleneck]
        dec_in = z.unsqueeze(1).repeat(1, self.W, 1)      # [B, W, bottleneck]
        h0 = self.from_latent(z).unsqueeze(0)             # [1, B, hidden]
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.dec(dec_in, (h0, c0))           # [B, W, hidden]
        return self.out(dec_out)                          # [B, W, C]


# =============================================================================
# Mapeo de canal dominante -> familia de anomalía (atribución, no clasificador)
# =============================================================================
def channel_family(col: str) -> str:
    """Traduce el canal que domina el error a una familia de anomalía legible.

    Es una heurística de atribución: indica DÓNDE vive la desviación, no
    clasifica el tipo exacto. Varios tipos sintéticos (spike, reversal,
    sign_flip) comparten los canales de net_movement y no se separan entre sí.
    """
    if "line_count" in col:
        return "volumen (line_count)"
    if "running_balance" in col:
        return "nivel (running_balance)"
    if "active_ou" in col:
        return "estructura OU (active_ou_count)"
    return "magnitud/temporal (net_movement)"


# =============================================================================
# Wrapper de detección
# =============================================================================
class LSTMAnomalyAE:
    def __init__(
        self,
        feature_cols: Sequence[str],
        series_keys: Sequence[str],
        period_date_col: str = "period_date",
        id_cols: Optional[Sequence[str]] = None,
        cfg: Optional[LSTMConfig] = None,
    ):
        self.feature_cols = list(feature_cols)
        self.series_keys = list(series_keys)
        self.period_date_col = period_date_col
        self.id_cols = list(id_cols) if id_cols else list(series_keys)
        self.cfg = cfg or LSTMConfig()
        self.model: Optional[_LSTMAE] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    def _seed(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "LSTMAnomalyAE":
        cfg = self.cfg
        self._seed()
        windows, _ = build_windows(X_train, self.feature_cols, self.series_keys,
                                   self.period_date_col, cfg.window)
        if len(windows) == 0:
            raise ValueError("No hay ventanas: revisa W frente a la longitud de las series.")
        data = torch.tensor(windows, device=self.device)
        n, W, C = data.shape

        perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
        n_val = max(1, int(n * cfg.val_fraction))
        X_val, X_tr = data[perm[:n_val]], data[perm[n_val:]]

        self.model = _LSTMAE(C, cfg).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        best_val, best_state, no_improve = float("inf"), None, 0
        self.history = {"train": [], "val": []}
        for _ in range(cfg.max_epochs):
            self.model.train()
            bperm = torch.randperm(X_tr.shape[0])
            tot = 0.0
            for s in range(0, X_tr.shape[0], cfg.batch_size):
                b = X_tr[bperm[s:s + cfg.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model(b), b)
                loss.backward()
                opt.step()
                tot += loss.item() * b.shape[0]
            tr = tot / X_tr.shape[0]
            self.model.eval()
            with torch.no_grad():
                vl = loss_fn(self.model(X_val), X_val).item()
            self.history["train"].append(tr)
            self.history["val"].append(vl)
            if vl < best_val - 1e-6:
                best_val, best_state, no_improve = vl, copy.deepcopy(self.model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ #
    def _recon_sq(self, windows: np.ndarray) -> np.ndarray:
        """Error cuadrático crudo por ventana, paso y canal: [n, W, C]."""
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(windows, device=self.device)
            recon = self.model(data)
            sq = ((recon - data) ** 2).cpu().numpy()
        return sq

    def _errors(self, windows: np.ndarray) -> np.ndarray:
        """Error total por ventana según score_mode (para el arnés de evaluación)."""
        sq = self._recon_sq(windows)
        if self.cfg.score_mode == "last":
            return sq[:, -1, :].mean(axis=1)               # error en el último período
        return sq.mean(axis=(1, 2))                        # error de toda la ventana

    def score_frame(self, X: pd.DataFrame, with_channels: bool = True) -> pd.DataFrame:
        """Puntúa cada (cuenta, período) y devuelve el ranking de candidatos.

        Con with_channels=True agrega el canal dominante del error y la familia
        de anomalía atribuida, útil para el reporte operativo top-K.
        """
        if self.model is None:
            raise RuntimeError("Llama a fit() antes de puntuar.")
        windows, meta = build_windows(X, self.feature_cols, self.series_keys,
                                      self.period_date_col, self.cfg.window)
        if len(windows) == 0:
            out = X[self.id_cols].iloc[0:0].copy(); out["recon_error"] = []
            return out

        sq = self._recon_sq(windows)
        if self.cfg.score_mode == "last":
            per_chan = sq[:, -1, :]                         # [n, C]
        else:
            per_chan = sq.mean(axis=1)                      # [n, C]
        total = per_chan.mean(axis=1)

        out = meta[self.id_cols].copy()
        out["recon_error"] = total
        if with_channels:
            top_idx = per_chan.argmax(axis=1)
            row_sum = per_chan.sum(axis=1)
            out["top_channel"] = [self.feature_cols[i] for i in top_idx]
            out["top_channel_share"] = np.round(
                per_chan[np.arange(len(per_chan)), top_idx] / np.where(row_sum == 0, 1, row_sum), 3)
            out["anomaly_family"] = [channel_family(self.feature_cols[i]) for i in top_idx]
        return out.sort_values("recon_error", ascending=False).reset_index(drop=True)


# =============================================================================
# Adaptador al arnés de eval_injection
# =============================================================================
def make_lstm_scorer(feature_builder, lstm_ae, series_keys, period_col):
    """Envuelve (FeatureBuilder ajustado + LSTMAnomalyAE entrenado) como scorer."""
    def scorer(df: pd.DataFrame) -> pd.DataFrame:
        X = feature_builder.transform(df)
        windows, meta = build_windows(X, lstm_ae.feature_cols, series_keys,
                                      lstm_ae.period_date_col, lstm_ae.cfg.window)
        res = meta[[*series_keys, period_col]].copy()
        res["score"] = lstm_ae._errors(windows) if len(windows) else []
        return res
    return scorer


# =============================================================================
# Demo (requiere torch; correr en tu entorno). Aquí solo se testea el windowing.
# =============================================================================
if __name__ == "__main__":
    from features_l2 import FeatureBuilder, FeatureConfig

    rng = np.random.default_rng(0)
    periods = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for acct in [f"acc_{k:02d}" for k in range(40)]:
        season = 5_000 * (1 + np.sin(2 * np.pi * periods.month / 12))
        vals = season + rng.normal(0, 800, len(periods))
        running = 0.0
        for d, v in zip(periods, vals):
            running += float(v)
            rows.append(dict(account=acct, ou_group="UNV", period_name=d.strftime("%b-%y"),
                             period_date=d, net_movement=float(v),
                             line_count=int(rng.integers(5, 50)), active_ou_count=2,
                             no_movement=0, running_balance=running))
    demo = pd.DataFrame(rows)

    cfg = FeatureConfig()
    fb = FeatureBuilder(cfg).fit(demo, selected_series=FeatureBuilder.select_dense_series(demo, cfg))
    X_all = fb.transform(demo)

    lae = LSTMAnomalyAE(feature_cols=fb.feature_cols, series_keys=["account"],
                        id_cols=["account", "period_name", "period_date", "net_movement"])
    lae.fit(X_train=X_all[X_all.period_date < "2024-07-01"])
    scored = lae.score_frame(X_all)
    print("Top 5 por error de reconstrucción (LSTM-AE):")
    print(scored.head(5).to_string(index=False))