"""
autoencoder.py
==============

Autoencoder denso (PyTorch) para detección de anomalías de Nivel 2.

Idea
----
El AE aprende a reconstruir el vector de features "normal" de cada
(cuenta, período). Lo que no encaja en el patrón aprendido se reconstruye
mal: el **error de reconstrucción** es el score de anomalía.

Por qué denso y pequeño
-----------------------
La población es modesta (~379 series densas, ~30 puntos c/u). Una red grande
memorizaría ruido. Por eso: cuello de botella de 3-4 dimensiones, dropout,
weight decay y early stopping. El objetivo no es capacidad, es un manifold
"normal" bien regularizado.

Interpretabilidad
-----------------
`score_frame` devuelve, además del error total, el **error por feature** y el
canal que más contribuye a cada score. Esto es clave: en la cuenta 14024 el
movimiento de 385M dispara el error en el canal `slog` / lags, no en
`resid_seasonal`.

El AE consume la matriz YA estandarizada que produce features_l2.FeatureBuilder.

Convención: documentación en español, código e identificadores en inglés.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


# =============================================================================
# Configuración
# =============================================================================
@dataclass
class AEConfig:
    """Hiperparámetros del autoencoder. Los defaults están calibrados para
    un dataset pequeño: red angosta, regularización fuerte, parada temprana."""

    hidden_dims: tuple[int, ...] = (16, 8)   # capas del encoder; el decoder es espejo
    bottleneck: int = 4                      # dimensión del cuello de botella
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-5               # L2, ayuda contra el sobreajuste
    max_epochs: int = 300
    patience: int = 20                       # early stopping sobre pérdida de validación
    batch_size: int = 128
    val_fraction: float = 0.15               # fracción de train reservada para validación
    seed: int = 42


# =============================================================================
# Red
# =============================================================================
def _mlp(dims: Sequence[int], dropout: float) -> nn.Sequential:
    """MLP con ReLU + Dropout entre capas; la última capa es lineal."""
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        is_last = i == len(dims) - 2
        if not is_last:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _DenseAE(nn.Module):
    def __init__(self, n_features: int, cfg: AEConfig):
        super().__init__()
        self.encoder = _mlp([n_features, *cfg.hidden_dims, cfg.bottleneck], cfg.dropout)
        self.decoder = _mlp([cfg.bottleneck, *reversed(cfg.hidden_dims), n_features], cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# =============================================================================
# Wrapper de detección
# =============================================================================
class AnomalyAE:
    """Autoencoder de anomalías sobre la matriz de features estandarizada.

    Uso típico
    ----------
        from features_l2 import FeatureBuilder, FeatureConfig
        from autoencoder import AnomalyAE, AEConfig

        # ... construir X_train, X_all con FeatureBuilder ...
        ae = AnomalyAE(feature_cols=fb.feature_cols,
                       id_cols=["account", "period_name", "period_date", "net_movement"])
        ae.fit(X_train)

        scored = ae.score_frame(X_all)      # una fila por (cuenta, período), ordenada
        flagged, thr = ae.flag(scored, quantile=0.99)
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        id_cols: Optional[Sequence[str]] = None,
        cfg: Optional[AEConfig] = None,
    ):
        self.feature_cols = list(feature_cols)
        self.id_cols = list(id_cols) if id_cols else []
        self.cfg = cfg or AEConfig()
        self.model: Optional[_DenseAE] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    # ------------------------------------------------------------------ #
    def _matrix(self, X) -> np.ndarray:
        """Extrae la matriz de features como float32."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols]
        return np.asarray(X, dtype=np.float32)

    def _seed(self) -> None:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #
    def fit(self, X_train) -> "AnomalyAE":
        cfg = self.cfg
        self._seed()

        data = torch.tensor(self._matrix(X_train), device=self.device)
        n, n_features = data.shape

        # Split interno train/validación para early stopping (aleatorio, sembrado)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
        n_val = max(1, int(n * cfg.val_fraction))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_tr, X_val = data[tr_idx], data[val_idx]

        self.model = _DenseAE(n_features, cfg).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        best_val, best_state, epochs_no_improve = float("inf"), None, 0
        self.history = {"train": [], "val": []}

        for epoch in range(cfg.max_epochs):
            # --- train ---
            self.model.train()
            batch_perm = torch.randperm(X_tr.shape[0])
            epoch_loss = 0.0
            for start in range(0, X_tr.shape[0], cfg.batch_size):
                batch = X_tr[batch_perm[start:start + cfg.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model(batch), batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= X_tr.shape[0]

            # --- validación ---
            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(X_val), X_val).item()

            self.history["train"].append(epoch_loss)
            self.history["val"].append(val_loss)

            # --- early stopping ---
            if val_loss < best_val - 1e-6:
                best_val, best_state = val_loss, copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Llama a fit() antes de puntuar.")

    def reconstruction_error(self, X, per_feature: bool = False) -> np.ndarray:
        """Error cuadrático de reconstrucción. Total por fila, o por feature."""
        self._check_fitted()
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(self._matrix(X), device=self.device)
            recon = self.model(data)
            sq = (recon - data) ** 2
        sq = sq.cpu().numpy()
        return sq if per_feature else sq.mean(axis=1)

    def encode(self, X) -> np.ndarray:
        """Representación del cuello de botella (útil para visualizar el latente)."""
        self._check_fitted()
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(self._matrix(X), device=self.device)
            z = self.model.encoder(data)
        return z.cpu().numpy()

    def score_frame(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve id_cols + recon_error + canal dominante, ordenado desc.

        `top_feature` = feature con mayor error de reconstrucción en esa fila.
        `top_feature_share` = qué fracción del error total aporta ese canal.
        """
        total = self.reconstruction_error(X_df, per_feature=False)
        per_feat = self.reconstruction_error(X_df, per_feature=True)

        top_idx = per_feat.argmax(axis=1)
        top_feature = [self.feature_cols[i] for i in top_idx]
        row_sum = per_feat.sum(axis=1)
        top_share = per_feat[np.arange(len(per_feat)), top_idx] / np.where(row_sum == 0, 1, row_sum)

        out = X_df[self.id_cols].copy() if self.id_cols else pd.DataFrame(index=X_df.index)
        out["recon_error"] = total
        out["top_feature"] = top_feature
        out["top_feature_share"] = np.round(top_share, 3)
        return out.sort_values("recon_error", ascending=False).reset_index(drop=True)

    def flag(self, scored: pd.DataFrame, quantile: float = 0.99):
        """Marca como anomalía las filas por encima del percentil dado del error."""
        thr = scored["recon_error"].quantile(quantile)
        out = scored.copy()
        out["is_anomaly"] = out["recon_error"] >= thr
        return out, float(thr)

    # ------------------------------------------------------------------ #
    # Persistencia (reproducibilidad de la tesis)
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        self._check_fitted()
        torch.save(
            {"state_dict": self.model.state_dict(),
             "feature_cols": self.feature_cols,
             "cfg": self.cfg.__dict__},
            path,
        )

    def load(self, path: str) -> "AnomalyAE":
        ckpt = torch.load(path, map_location=self.device)
        self.feature_cols = ckpt["feature_cols"]
        self.cfg = AEConfig(**ckpt["cfg"])
        self.model = _DenseAE(len(self.feature_cols), self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        return self


# =============================================================================
# Demo (requiere torch)
# =============================================================================
if __name__ == "__main__":
    from features_L2 import FeatureBuilder, FeatureConfig

    # --- datos sintéticos con una reversión inyectada (imita a la 14024) ---
    rng = np.random.default_rng(0)
    periods = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for acct in [f"acc_{k:02d}" for k in range(40)]:
        season = 5_000 * (1 + np.sin(2 * np.pi * periods.month / 12))
        vals = season + rng.normal(0, 800, len(periods))
        running = 0.0
        for d, v in zip(periods, vals):
            running += float(v)
            rows.append(dict(account=acct, ou_group="UNV",
                             period_name=d.strftime("%b-%y"), period_date=d,
                             net_movement=float(v), line_count=int(rng.integers(5, 50)),
                             active_ou_count=int(rng.integers(1, 6)), no_movement=0,
                             running_balance=running))
    demo = pd.DataFrame(rows)

    # Inyectar una reversión de gran magnitud en una cuenta (abr/may 2024)
    a = (demo.account == "acc_07")
    demo.loc[a & (demo.period_name == "Apr-24"), "net_movement"] = -3.8e8
    demo.loc[a & (demo.period_name == "May-24"), "net_movement"] = +3.8e8

    cfg = FeatureConfig()
    fb = FeatureBuilder(cfg)
    split = pd.Timestamp("2025-07-01")
    dense = FeatureBuilder.select_dense_series(demo, cfg)
    fb.fit(demo[demo.period_date < split], selected_series=dense)
    X_all = fb.transform(demo)
    X_train = X_all[X_all.period_date < split]

    ae = AnomalyAE(feature_cols=fb.feature_cols,
                   id_cols=["account", "period_name", "period_date", "net_movement"])
    ae.fit(X_train)

    scored = ae.score_frame(X_all)
    print("Top 5 anomalías por error de reconstrucción:")
    print(scored.head(5).to_string(index=False))
    print("\n¿La reversión inyectada (acc_07, abr/may-24) quedó arriba?")
    print(scored[scored.account == "acc_07"].head(3).to_string(index=False))