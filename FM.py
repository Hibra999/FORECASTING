# -*- coding: utf-8 -*-
# App mínima: una gráfica principal + pestañas "Evaluación" y "Tokens disponibles"
# Requisitos:
#   Python 3.11+
#   pip install "timesfm[torch]" yfinance plotly streamlit pandas numpy

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Forecast con TimesFM 2.0", page_icon="📈", layout="centered")
st.title("Predecir una acción (TimesFM 2.0)")

# ---- Controles mínimos ----
accion = st.text_input("Acción / Token (Ticker de Yahoo Finance)", "AAPL")
col1, col2 = st.columns(2)
with col1:
    periodo = st.selectbox("Periodo histórico", ["6mo","1y","2y","5y","10y","max"], index=2)
with col2:
    dias = st.slider("Días a predecir", 5, 365, 30)

boton1 = st.button("Predecir")

# ---- Import seguro de TimesFM ----
def _import_timesfm():
    try:
        import torch
        from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
        return torch, TimesFm, TimesFmHparams, TimesFmCheckpoint
    except Exception:
        st.error(
            "No pude importar **TimesFM**. Instala y vuelve a ejecutar:\n\n"
            "```bash\npip install \"timesfm[torch]\" yfinance plotly streamlit\n```"
        )
        st.stop()

# ---- Utilidades mínimas ----
def _infer_freq(idx: pd.DatetimeIndex, prefer_business=True) -> str:
    if idx is None or len(idx) < 3: return "D"
    inferred = pd.infer_freq(idx)
    if inferred is None: return "B" if prefer_business else "D"
    for k in ("B","D","W","M","Q","A","Y"):
        if inferred.startswith(k): return ("Y" if k in ("A","Y") else k)
    return "D"

def _extract_close_series(data: pd.DataFrame, tkr: str) -> pd.Series:
    df = data.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close","Adj Close","close","adj close"]:
            if cand in df.columns: return pd.to_numeric(df[cand], errors="coerce")
        lower = {c.lower(): c for c in df.columns}
        if "close" in lower:  return pd.to_numeric(df[lower["close"]], errors="coerce")
        if "adj close" in lower: return pd.to_numeric(df[lower["adj close"]], errors="coerce")
        raise KeyError("No se encontró columna Close/Adj Close en los datos.")
    # MultiIndex
    def has_lv(level_idx, name):
        try: return name in df.columns.get_level_values(level_idx)
        except Exception: return False
    if has_lv(-1,"Close") or has_lv(-1,"Adj Close"):
        try: sub = df.xs("Close", level=-1, axis=1)
        except KeyError: sub = df.xs("Adj Close", level=-1, axis=1)
        sel = tkr if tkr in sub.columns else sub.columns[0]
        return pd.to_numeric(sub[sel], errors="coerce")
    if has_lv(0,"Close") or has_lv(0,"Adj Close"):
        try: sub = df.xs("Close", level=0, axis=1)
        except KeyError: sub = df.xs("Adj Close", level=0, axis=1)
        sel = tkr if tkr in sub.columns else sub.columns[0]
        return pd.to_numeric(sub[sel], errors="coerce")
    raise KeyError("No se pudo localizar la columna de cierre (MultiIndex).")

def _prepare_df_yf(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    s = _extract_close_series(data, ticker).astype(np.float32)
    ds = pd.to_datetime(s.index).tz_localize(None)
    out = pd.DataFrame({"unique_id": str(ticker), "ds": ds, "y": s.values})
    return out.dropna().sort_values("ds").reset_index(drop=True)

def _ceil_to(x: int, base: int) -> int:
    return int(np.ceil(max(1, x) / float(base)) * base)

def _build_tfm(horizon: int, prefer_gpu: bool, bs: int = 32):
    torch, TimesFm, TimesFmHparams, TimesFmCheckpoint = _import_timesfm()
    backend = "gpu" if (prefer_gpu and torch.cuda.is_available()) else "cpu"
    horizon_eff = _ceil_to(horizon, 128)  # decodificador trabaja en bloques de 128
    hparams = TimesFmHparams(
        backend=backend,
        per_core_batch_size=int(bs),
        # Parámetros fijos del checkpoint 2.0-500m
        input_patch_len=32,
        output_patch_len=128,
        num_layers=50,
        model_dims=1280,
        use_positional_embedding=False,
        # Generales
        context_len=2048,
        horizon_len=horizon_eff,
    )
    ckpt = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
    return TimesFm(hparams=hparams, checkpoint=ckpt)

def _pick_pred_col(fdf: pd.DataFrame) -> str:
    lc = [c.lower() for c in fdf.columns]
    for target in ["timesfm-q-0.5","timesfm","yhat","forecast","mean"]:
        for i,c in enumerate(lc):
            if c == target: return fdf.columns[i]
    meds = [c for c in fdf.columns if c.lower().endswith("q-0.5")]
    return meds[0] if meds else fdf.columns[-1]

def _plot(hist_df: pd.DataFrame, fc_df: pd.DataFrame, pred_col: str, titulo: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["ds"], y=hist_df["y"], name="Reales", mode="lines"))
    fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df[pred_col], name="Predicción (TimesFM)", mode="lines"))
    fig.update_layout(title=titulo, xaxis_title="Fecha", yaxis_title="Precio de cierre", height=520,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig

def _r2_score(y_true, y_pred) -> float:
    try:
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))
    except Exception:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.nansum((y_true - y_pred) ** 2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def _plot_eval(test_df: pd.DataFrame, pred_df: pd.DataFrame, pred_col: str, titulo: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["ds"], y=test_df["y"], name="Real (test)", mode="lines"))
    fig.add_trace(go.Scatter(x=pred_df["ds"], y=pred_df[pred_col], name="Predicción", mode="lines"))
    fig.update_layout(title=titulo, xaxis_title="Fecha", yaxis_title="Precio de cierre", height=520,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig

# ---- Datos de ejemplo para "Tokens disponibles" ----
TOKENS_CRYPTO = [
    ("BTC-USD","Bitcoin"),("ETH-USD","Ethereum"),("BNB-USD","BNB"),
    ("SOL-USD","Solana"),("XRP-USD","XRP"),("DOGE-USD","Dogecoin"),
    ("ADA-USD","Cardano"),("TON-USD","Toncoin"),("DOT-USD","Polkadot"),
    ("AVAX-USD","Avalanche")
]
ACCIONES_EJEMPLO = [
    ("AAPL","Apple"),("MSFT","Microsoft"),("NVDA","NVIDIA"),
    ("AMZN","Amazon"),("GOOGL","Alphabet"),("META","Meta"),
    ("TSLA","Tesla"),("NFLX","Netflix"),("AMD","AMD"),("INTC","Intel")
]

# ---- Flujo mínimo ----
if boton1:
    # 1) Descarga
    data = yf.download(
        accion, period=periodo, progress=False,
        group_by="column", auto_adjust=False, threads=True
    )
    if data is None or data.empty:
        st.error("No hay datos para ese ticker/periodo.")
        st.stop()

    # 2) Preparación
    try:
        input_df = _prepare_df_yf(data, accion)
    except Exception as e:
        st.error(f"Error preparando datos históricos: {e}")
        st.stop()

    freq = _infer_freq(data.index)  # auto
    tfm = _build_tfm(dias, prefer_gpu=True, bs=32)

    # 3) Pronóstico principal (gráfica principal)
    try:
        fc_df = tfm.forecast_on_df(inputs=input_df, freq=freq, value_name="y", num_jobs=-1)
        if "ds" in fc_df.columns: fc_df = fc_df.sort_values("ds")
        fc_df = fc_df.iloc[:dias].copy()  # recorte al horizonte solicitado
        pred_col = _pick_pred_col(fc_df)
    except Exception as e:
        st.error(f"Ocurrió un error al pronosticar con TimesFM: {e}")
        st.stop()

    # ---- TABS: Gráfica | Evaluación | Tokens ----
    tab_graf, tab_eval, tab_tokens = st.tabs(["📈 Gráfica", "🧪 Evaluación", "🪙 Tokens disponibles"])

    # 4) ÚNICA SALIDA PRINCIPAL: LA GRÁFICA
    with tab_graf:
        fig = _plot(input_df, fc_df, pred_col, f"{accion} — TimesFM 2.0 (500M)")
        st.plotly_chart(fig, use_container_width=True)

    # 5) Evaluación (train = todo menos últimos 'dias', test = últimos 'dias')
    with tab_eval:
        if len(input_df) <= dias + 5:
            st.warning("Muy pocos datos para evaluar: aumenta el periodo histórico o reduce 'Días a predecir'.")
        else:
            train_df = input_df.iloc[:-dias].copy()
            test_df  = input_df.iloc[-dias:].copy()

            try:
                tfm_eval = _build_tfm(dias, prefer_gpu=True, bs=32)
                fc_eval = tfm_eval.forecast_on_df(inputs=train_df, freq=freq, value_name="y", num_jobs=-1)
                if "ds" in fc_eval.columns: fc_eval = fc_eval.sort_values("ds")
                fc_eval = fc_eval.iloc[:dias].copy()
                pred_col_eval = _pick_pred_col(fc_eval)

                # Alinear por 'ds'; si falla, alinear por posición
                merged = pd.merge(test_df[["ds","y"]], fc_eval[["ds", pred_col_eval]],
                                  on="ds", how="inner")
                if merged.empty:
                    # fallback posicional
                    m = min(len(test_df), len(fc_eval))
                    merged = pd.DataFrame({
                        "ds": test_df["ds"].iloc[:m].values,
                        "y":  test_df["y"].iloc[:m].values,
                        pred_col_eval: fc_eval[pred_col_eval].iloc[:m].values
                    })

                r2 = _r2_score(merged["y"].values, merged[pred_col_eval].values)
                st.metric("R² en ventana de prueba", f"{r2:.4f}")

                fig_eval = _plot_eval(
                    test_df=merged.rename(columns={"y":"y"}),
                    pred_df=merged.rename(columns={pred_col_eval: pred_col_eval}),
                    pred_col=pred_col_eval,
                    titulo=f"Evaluación (últimos {dias} días) — {accion}"
                )
                st.plotly_chart(fig_eval, use_container_width=True)
            except Exception as e:
                st.error(f"Error en la evaluación: {e}")

    # 6) Tokens disponibles
    with tab_tokens:
        st.write("### Cripto (Yahoo Finance)")
        st.dataframe(pd.DataFrame(TOKENS_CRYPTO, columns=["Ticker","Nombre"]), use_container_width=True, hide_index=True)
        st.write("### Acciones de ejemplo")
        st.dataframe(pd.DataFrame(ACCIONES_EJEMPLO, columns=["Ticker","Nombre"]), use_container_width=True, hide_index=True)
