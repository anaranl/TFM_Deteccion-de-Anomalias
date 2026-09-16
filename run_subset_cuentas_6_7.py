"""
run_subset_cuentas_6_7.py
==========================

Corrida de robust_z (baseline), Autoencoder denso y LSTM-AE sobre el
subconjunto de cuentas que empiezan por 6 y 7, usando la interseccion de
las 283 cuentas con historia suficiente para los tres modelos (umbral mas
restrictivo: LSTM-AE, ventana W=12).

Ajustado al codigo real de: db.py, features_l2.py, autoencoder.py,
lstm_autoencoder.py y baseline.py.
"""

import pandas as pd
import numpy as np

from db import get_connection
from features_L2 import FeatureBuilder, FeatureConfig
from Autoencoder import AnomalyAE, AEConfig
from lstm_autoencoder import LSTMAnomalyAE, LSTMConfig, make_lstm_scorer
from baseline import robust_zscore, MATERIALITY_USD
from eval_injection import evaluate, InjectionConfig


# -----------------------------------------------------------------------
# 1. Cargar datos filtrados por cuenta (6 y 7) directamente desde SQL
# -----------------------------------------------------------------------
query = """
SELECT *
FROM notif.vw_ML_L2_CompleteSeries
WHERE LEFT(CAST(account AS varchar(20)), 1) IN ('6', '7')
"""

conn = get_connection(env="PROD")
df_clean = pd.read_sql(query, conn)
df_clean["period_date"] = pd.to_datetime(df_clean["period_date"])

print(f"Filas cargadas: {len(df_clean)}")
print(f"Cuentas distintas: {df_clean['account'].nunique()}")


# -----------------------------------------------------------------------
# 2. Determinar las 283 cuentas validas para LSTM-AE (W=12), el umbral
#    mas restrictivo, para que los 3 modelos usen exactamente la misma
#    poblacion
# -----------------------------------------------------------------------
periodos_por_cuenta = df_clean.groupby("account").size()
UMBRAL_LSTM = 13  # W=12 + al menos 1 periodo para puntuar

cuentas_comunes = periodos_por_cuenta[periodos_por_cuenta >= UMBRAL_LSTM].index
n_cuentas = len(cuentas_comunes)
print(f"Cuentas comunes a los 3 modelos: {n_cuentas}")
assert n_cuentas == 283, f"Se esperaban 283 y se obtuvieron {n_cuentas}; revisa si los datos cambiaron."

# FeatureBuilder.select_dense_series() trabaja con tuplas de series_keys;
# con series_keys=["account"] cada tupla es (account,)
cuentas_comunes_tuples = {(acct,) for acct in cuentas_comunes}

df_final = df_clean[df_clean["account"].isin(cuentas_comunes)].copy()


# -----------------------------------------------------------------------
# 3. Split temporal + fit del FeatureBuilder SOLO con train
#  
# -----------------------------------------------------------------------
MESES_TEST = 6   # mismo criterio usado con la poblacion completa
ultimo_periodo = df_final["period_date"].max()
split = ultimo_periodo - pd.DateOffset(months=MESES_TEST)
print(f"Fecha de corte train/test: {split} (ultimos {MESES_TEST} meses como test)")

feat_cfg = FeatureConfig()   # defaults ya apuntan a las columnas reales de la vista

fb = FeatureBuilder(feat_cfg)
fb.fit(df_final[df_final["period_date"] < split], selected_series=cuentas_comunes_tuples)

X_all = fb.transform(df_final)
X_train = X_all[X_all["period_date"] < split]
X_test = X_all[X_all["period_date"] >= split]

print(f"Cuentas tras fit/transform: {X_all['account'].nunique()}")
print(f"Train: {len(X_train)} filas | Test: {len(X_test)} filas")
print(f"Columnas de features: {fb.feature_cols}")


# -----------------------------------------------------------------------
# 4. Entrenar Autoencoder denso
# -----------------------------------------------------------------------
ae = AnomalyAE(
    feature_cols=fb.feature_cols,
    id_cols=["account", "period_name", "period_date", "net_movement"],
    cfg=AEConfig(),   # defaults ya calibrados (bottleneck=4, dropout, early stopping)
)
ae.fit(X_train)


# -----------------------------------------------------------------------
# 5. Entrenar LSTM-AE (EncDec-AD, ventana W=12, score en el ultimo periodo)
# -----------------------------------------------------------------------
lstm_ae = LSTMAnomalyAE(
    feature_cols=fb.feature_cols,
    series_keys=["account"],
    period_date_col="period_date",
    id_cols=["account", "period_name", "period_date", "net_movement"],
    cfg=LSTMConfig(),   # defaults: window=12, hidden=32, bottleneck=8, score_mode="last"
)
lstm_ae.fit(X_train)


# -----------------------------------------------------------------------
# 6. Definir los 3 scorers bajo el contrato que espera eval_injection:
#    un DataFrame con [account, period_name, score]
# -----------------------------------------------------------------------
def scorer_robust_z(df):
    scored = robust_zscore(df, min_periods=12, materiality=MATERIALITY_USD)
    if scored.empty:
        return pd.DataFrame(columns=["account", "period_name", "score"])
    out = scored[["account", "period_name"]].copy()
    out["score"] = scored["abs_score_z"]
    return out

def scorer_ae(df):
    Xt = fb.transform(df)                     # re-featurizar sin reajustar
    scored = ae.score_frame(Xt)               # ya trae account, period_name, recon_error
    return scored.rename(columns={"recon_error": "score"})[["account", "period_name", "score"]]

# El LSTM ya trae su propio adaptador listo (une FeatureBuilder + LSTMAnomalyAE)
scorer_lstm = make_lstm_scorer(fb, lstm_ae, series_keys=["account"], period_col="period_name")


# -----------------------------------------------------------------------
# 7. Evaluar los 3 detectores con el mismo arnes de inyeccion sintetica,
#    sobre el mismo df_final (283 cuentas)
# -----------------------------------------------------------------------
cfg = InjectionConfig(n_injections=60, n_repeats=5)

res_robust_z = evaluate(df_final, scorer_robust_z, cfg, series_keys=["account"],
                        detector_name="robust_z")
res_ae = evaluate(df_final, scorer_ae, cfg, series_keys=["account"],
                  detector_name="AE_denso")
res_lstm = evaluate(df_final, scorer_lstm, cfg, series_keys=["account"],
                    detector_name="LSTM-AE")

resultados = pd.concat([res_robust_z, res_ae, res_lstm], ignore_index=True)
print("\nResultados comparativos (cuentas 6 y 7, n=283):\n")
print(resultados.to_string(index=False))

resultados.to_csv("resultados_subset_cuentas_6_7.csv", index=False)
print("\nGuardado en resultados_subset_cuentas_6_7.csv")
