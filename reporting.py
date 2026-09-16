"""
reporting.py
============

Exportación de candidatos de anomalía del detector de Nivel 2.

Opera sobre el DataFrame que devuelve LSTMAnomalyAE.score_frame (o
AnomalyAE.score_frame), no sobre el modelo, así que no requiere torch.

Produce:
  - CSV con el top-K global de candidatos.
  - CSV con el top-K dentro de cada familia de anomalía.
  - Una tabla formateada (DataFrame) lista para mostrar o pegar en el anexo.
  - Una versión en Markdown de la tabla.

Los candidatos son señales de triage para revisión de Finanzas, no anomalías
confirmadas.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import pandas as pd

# Columnas de salida y sus nombres legibles para el anexo.
_RENAME = {
    "account": "Cuenta",
    "period_name": "Período",
    "net_movement": "Movimiento neto (USD)",
    "recon_error": "Error recon.",
    "top_channel": "Canal dominante",
    "top_channel_share": "Peso canal",
    "anomaly_family": "Familia",
}
_ORDER = ["account", "period_name", "net_movement", "recon_error",
          "top_channel", "top_channel_share", "anomaly_family"]


def candidates_table(report: pd.DataFrame, k: int = 10,
                     family: Optional[str] = None, rename: bool = True) -> pd.DataFrame:
    """Top-K candidatos como tabla ordenada y formateada.

    - family: si se indica, filtra a esa familia de anomalía antes del top-K.
    - rename: usa encabezados legibles en español.
    """
    df = report.copy()
    if family is not None:
        if "anomaly_family" not in df.columns:
            raise KeyError("El report no tiene 'anomaly_family'. "
                           "Genera score_frame con with_channels=True.")
        df = df[df["anomaly_family"] == family]

    df = df.sort_values("recon_error", ascending=False).head(k).reset_index(drop=True)

    cols = [c for c in _ORDER if c in df.columns]
    df = df[cols].copy()

    # Redondeos y formato de presentación
    if "recon_error" in df:
        df["recon_error"] = df["recon_error"].round(3)
    if "top_channel_share" in df:
        df["top_channel_share"] = df["top_channel_share"].round(3)
    if "net_movement" in df:
        df["net_movement"] = df["net_movement"].round(2)

    df.insert(0, "rank", range(1, len(df) + 1))
    if rename:
        df = df.rename(columns={**_RENAME, "rank": "#"})
    return df


def export_topk_csv(report: pd.DataFrame, path: str, k: int = 10) -> str:
    """Escribe el top-K global de candidatos a un CSV (UTF-8 con BOM para Excel)."""
    tabla = candidates_table(report, k=k, rename=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tabla.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_by_family_csv(report: pd.DataFrame, path: str, k: int = 10) -> str:
    """Escribe el top-K de cada familia de anomalía, apilado, a un solo CSV."""
    if "anomaly_family" not in report.columns:
        raise KeyError("El report no tiene 'anomaly_family'. "
                       "Genera score_frame con with_channels=True.")
    partes = []
    for fam in sorted(report["anomaly_family"].dropna().unique()):
        partes.append(candidates_table(report, k=k, family=fam, rename=True))
    out = pd.concat(partes, ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def to_markdown(report: pd.DataFrame, k: int = 10, family: Optional[str] = None) -> str:
    """Top-K como tabla Markdown (para pegar en notas o documentos)."""
    tabla = candidates_table(report, k=k, family=family, rename=True)
    try:
        return tabla.to_markdown(index=False)
    except Exception:
        # Fallback si falta 'tabulate': markdown manual
        cols = list(tabla.columns)
        lines = ["| " + " | ".join(cols) + " |",
                 "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in tabla.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)


def export_candidates(report: pd.DataFrame, out_dir: str = "reports",
                      k: int = 10, prefix: str = "candidatos_L2") -> dict:
    """Exporta ambos CSV (global y por familia) y devuelve las rutas."""
    global_path = os.path.join(out_dir, f"{prefix}_top{k}.csv")
    family_path = os.path.join(out_dir, f"{prefix}_por_familia_top{k}.csv")
    paths = {"global": export_topk_csv(report, global_path, k=k)}
    if "anomaly_family" in report.columns:
        paths["por_familia"] = export_by_family_csv(report, family_path, k=k)
    return paths


# =============================================================================
# Prueba con un report sintético (sin torch)
# =============================================================================
if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    fams = ["magnitud/temporal (net_movement)", "volumen (line_count)",
            "nivel (running_balance)", "estructura OU (active_ou_count)"]
    chans = {"magnitud/temporal (net_movement)": "slog",
             "volumen (line_count)": "resid_line_count",
             "nivel (running_balance)": "resid_running_balance",
             "estructura OU (active_ou_count)": "resid_active_ou_count"}
    rows = []
    for i in range(40):
        fam = rng.choice(fams)
        rows.append(dict(
            account=f"{rng.integers(10000, 99999)}",
            period_name=f"{rng.choice(['Jan','Apr','Jul','Oct'])}-25",
            period_date=pd.Timestamp("2025-01-01"),
            net_movement=float(rng.normal(0, 5e5)),
            recon_error=float(abs(rng.normal(3, 2))),
            top_channel=chans[fam],
            top_channel_share=float(rng.uniform(0.4, 0.95)),
            anomaly_family=fam,
        ))
    report = pd.DataFrame(rows)

    print("=== Top-10 global ===")
    print(candidates_table(report, k=10).to_string(index=False))
    paths = export_candidates(report, out_dir="/tmp/rep_demo", k=10)
    print("\nCSV escritos:", paths)
    print("\n=== Markdown (top 5) ===")
    print(to_markdown(report, k=5))
