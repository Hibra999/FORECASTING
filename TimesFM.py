# Reemplazo de Prophet por TimesFM (v2.0 500M) con opción de fine-tuning
# NOTA: Instala primero:
#   pip install "timesfm[torch]" yfinance plotly streamlit
# Requiere Python 3.11+ para la versión Torch de TimesFM.
# (Checkpoint usado: google/timesfm-2.0-500m-pytorch)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf

# ---- INTENTO DE CARGA DE TIMESFM (Torch) ----
try:
    import torch
    import timesfm
    from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
except Exception as e:
    st.error(
        "No pude importar TimesFM. Instala los paquetes y vuelve a ejecutar:\n\n"
        "    pip install 'timesfm[torch]'"
    )
    st.stop()

st.title("Predecir una acción con TimesFM 2.0 (500M)")

# --------- UI ---------
col1, col2 = st.columns(2)
with col1:
    accion = st.text_input("Ticker (Yahoo Finance)", "AAPL")
with col2:
    dias = st.slider("Días a predecir (horizonte)", 5, 365, 30)

adv = st.expander("Opciones avanzadas (robustas)")
with adv:
    freq_manual = st.selectbox(
        "Frecuencia (si no estás seguro, deja 'auto')",
        options=["auto", "B (laborales)", "D (diaria)", "W", "M", "Q", "Y"],
        index=0,
    )
    usar_gpu = st.checkbox("Forzar GPU si está disponible", value=True)
    bs = st.number_input("Batch size por núcleo", min_value=1, max_value=64, value=32)

    st.markdown("---")
    st.subheader("Fine-tuning (opcional)")
    ft_enable = st.checkbox("Hacer fine-tuning (PEFT/adapter) con mis datos", value=False)
    ft_data = st.file_uploader(
        "Sube CSV con columnas: ds (fecha), y (valor). Opcional: unique_id para múltiples series.",
        type=["csv"],
    )
    ft_epochs = st.number_input("Épocas", 1, 50, 5)
    ft_lr = st.number_input("Learning rate", 1e-6, 1e-2, 5e-4, format="%.6f")
    ft_output_dir = st.text_input("Carpeta para guardar el adapter", "timesfm_adapter")

boton1 = st.button("Predecir")

# --------- Utilidades ---------
def _infer_freq(idx: pd.DatetimeIndex, prefer_business=True):
    """Intenta inferir la frecuencia adecuada para TimesFM."""
    if idx is None or len(idx) < 3:
        return "D"
    # pandas puede devolver None; manejamos valores típicos de equity
    inferred = pd.infer_freq(idx)
    if inferred is None:
        # Si son días bursátiles (faltan fines de semana)
        return "B" if prefer_business else "D"
    # Normaliza a códigos esperados por TimesFM
    if inferred.startswith("B"):
        return "B"
    if inferred.startswith("D"):
        return "D"
    if inferred.startswith("W"):
        return "W"
    if inferred.startswith("M"):
        return "M"
    if inferred.startswith("Q"):
        return "Q"
    if inferred.startswith("A") or inferred.startswith("Y"):
        return "Y"
    return "D"

def _build_tfm(horizon: int, backend_pref_gpu: bool, per_core_bs: int):
    """Inicializa TimesFM 2.0 500M (Torch)."""
    backend = "gpu" if (backend_pref_gpu and torch.cuda.is_available()) else "cpu"
    # Hparams recomendados para el checkpoint 2.0 (500M)
    hparams = TimesFmHparams(
        backend=backend,
        per_core_batch_size=int(per_core_bs),
        horizon_len=int(horizon),
        num_layers=50,
        context_len=2048,              # máx. contexto del modelo 2.0
        use_positional_embedding=False # según README del checkpoint 2.0
    )
    checkpoint = TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    )
    return TimesFm(hparams=hparams, checkpoint=checkpoint)

def _prepare_df_yf(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convierte datos de yfinance -> formato TimesFM: unique_id, ds, y."""
    df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df["unique_id"] = ticker
    # Orden y tipos
    df = df[["unique_id", "ds", "y"]].dropna().sort_values("ds")
    return df

def _pick_pred_column(fdf: pd.DataFrame) -> str:
    """TimesFM suele devolver 'timesfm' o cuantiles 'timesfm-q-0.5'."""
    lc = [c.lower() for c in fdf.columns]
    # Prioridad: mediana si existe
    for target in ["timesfm-q-0.5", "timesfm", "yhat", "forecast", "mean"]:
        for i, c in enumerate(lc):
            if c == target:
                return fdf.columns[i]
    # Si trae varios cuantiles, intenta la mediana por patrón
    med = [c for c in fdf.columns if c.lower().endswith("q-0.5")]
    return med[0] if med else fdf.columns[-1]  # fallback conservador

def _plot_forecast(hist_df: pd.DataFrame, fc_df: pd.DataFrame, pred_col: str, titulo: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["y"], name="Reales", mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["ds"], y=fc_df[pred_col], name="Predicción (TimesFM)", mode="lines"
    ))
    fig.update_layout(
        title=titulo,
        xaxis_title="Fecha",
        yaxis_title="Precio de cierre",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=500,
    )
    return fig

# --------- Acción principal ---------
if boton1:
    # 1) Descarga de datos (2 años)
    data = yf.download(accion, period="2y", progress=False)
    if data is None or data.empty:
        st.error("No hay datos para ese ticker.")
        st.stop()

    st.line_chart(data["Close"])
    st.dataframe(data.tail(5))

    # 2) Prepara datos para TimesFM
    input_df = _prepare_df_yf(data, accion)

    # Frecuencia
    if freq_manual == "auto":
        freq = _infer_freq(data.index)
    else:
        freq = freq_manual.split()[0]  # toma código antes del paréntesis

    # 3) Opcional: Fine-tuning (adapter)
    adapter_loaded = False
    if ft_enable and ft_data is not None:
        try:
            user_df = pd.read_csv(ft_data)
            # Normaliza columnas
            cols = {c.lower(): c for c in user_df.columns}
            # exige ds, y; unique_id opcional
            if "ds" not in cols or "y" not in cols:
                raise ValueError("El CSV debe tener columnas 'ds' y 'y'.")
            if "unique_id" not in cols:
                user_df["unique_id"] = accion
                cols["unique_id"] = "unique_id"
            # casteos
            user_df = user_df.rename(
                columns={cols["ds"]: "ds", cols["y"]: "y", cols["unique_id"]: "unique_id"}
            )
            user_df["ds"] = pd.to_datetime(user_df["ds"])
            user_df = user_df[["unique_id", "ds", "y"]].dropna().sort_values(["unique_id", "ds"])

            # Entrenamiento PEFT (siempre en try/except para no romper la app)
            # La API de finetuning oficial vive en notebooks/peft; aquí invocamos si está disponible.
            try:
                # Disponible desde la versión con soporte de finetuning (Torch)
                from timesfm.peft.finetuning_torch import TimesFMFinetuner, FinetuningConfig  # type: ignore

                # Config (valores razonables para adapters en series financieras)
                cfg = FinetuningConfig(
                    learning_rate=float(ft_lr),
                    epochs=int(ft_epochs),
                    output_dir=ft_output_dir,
                    save_every_epoch=False,
                    lora_rank=16,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    weight_decay=0.0,
                    gradient_accumulation_steps=1,
                    train_val_split=0.9,
                    seed=42,
                )

                # Construye finetuner (carga base 2.0 500M)
                finetuner = TimesFMFinetuner(
                    base_checkpoint_id="google/timesfm-2.0-500m-pytorch",
                    backend="gpu" if (usar_gpu and torch.cuda.is_available()) else "cpu",
                    per_core_batch_size=int(bs),
                )

                # Ejecuta fine-tuning
                finetuner.fit_on_dataframe(
                    df=user_df,
                    freq=freq,
                    value_name="y",
                    config=cfg,
                )

                # Después del entrenamiento, recarga el modelo base y aplica el adapter guardado
                tfm = _build_tfm(dias, usar_gpu, bs)
                finetuner.apply_adapter(tfm, adapter_dir=ft_output_dir)  # inyecta LoRA
                adapter_loaded = True
                st.success(f"Adapter cargado desde '{ft_output_dir}'.")
            except Exception as e:
                st.warning(
                    "No se pudo ejecutar el fine-tuning dentro de la app. "
                    "Asegúrate de usar la versión más reciente de `timesfm` con soporte PEFT. "
                    "Como alternativa, entrena el adapter fuera de Streamlit (notebook oficial) "
                    "y luego cárgalo aquí. Seguimos con el modelo base sin fine-tuning."
                )
                adapter_loaded = False
        except Exception as e:
            st.error(f"Error leyendo/entrenando con tu CSV de fine-tuning: {e}")
            adapter_loaded = False

    # 4) Inicializa el modelo (si no se cargó un adapter ya acoplado)
    if not adapter_loaded:
        tfm = _build_tfm(dias, usar_gpu, bs)

    # 5) Pronóstico
    try:
        # TimesFM espera 'unique_id', 'ds', 'y'
        fc_df = tfm.forecast_on_df(
            inputs=input_df,
            freq=freq,        # ej. "B" o "D"
            value_name="y",
            num_jobs=-1,
        )
        # Asegura horizonte requerido si el modelo devolvió más
        if "ds" in fc_df.columns:
            fc_df = fc_df.sort_values("ds")
        if len(fc_df) > dias:
            fc_df = fc_df.iloc[:dias].copy()

        pred_col = _pick_pred_column(fc_df)

        # 6) Mostrar resultados
        st.write(f"Frecuencia usada: **{freq}** | Backend: **{'GPU' if (usar_gpu and torch.cuda.is_available()) else 'CPU'}**")
        fig = _plot_forecast(input_df, fc_df, pred_col, f"{accion} — TimesFM 2.0 (500M)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fc_df.head(10))
    except Exception as e:
        st.error(f"Ocurrió un error al pronosticar con TimesFM: {e}")
