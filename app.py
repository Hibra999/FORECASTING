import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from prophet import Prophet

st.title("Predecir una accion")

accion = st.text_input("Accion", "AAPL")
dias = st.slider("Dias a predecir", 30,365)
boton1 = st.button("Predecir")

if boton1:
    data = yf.download(accion, period="2y")
    if data.empty:
        st.error("No hay datos")
    st.line_chart(data["Close"])
    st.dataframe(data.tail(5))
    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=dias)
    forecast = model.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df["ds"], y = df["y"], name = "reales"))
    fig.add_trace(go.Scatter(x = forecast["ds"], y = forecast["yhat"], name = "predicciones"))
    st.plotly_chart(fig)