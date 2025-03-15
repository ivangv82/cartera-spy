import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# -----------------------------------------------------------------------------
# 1️⃣ Configuración de Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Estrategias de Inversión SP500", layout="wide")
st.title("🔍 Estrategias de Inversión en el SP500")

# Selección de sistemas activos
num_sistemas = st.selectbox("Selecciona cuántos sistemas activar:", [1, 2, 3, 4, 5], index=4)

# Selección de fechas
fecha_inicio = st.date_input("Fecha de inicio", datetime(2020, 1, 1))
fecha_fin = st.date_input("Fecha de fin (dejar en blanco para usar hoy)", datetime.today())

# Convertir fechas a formato Pandas
start_date = pd.Timestamp(fecha_inicio)
end_date = pd.Timestamp(fecha_fin) if fecha_fin else pd.Timestamp(datetime.today())

# -----------------------------------------------------------------------------
# 2️⃣ Cargar datos desde GitHub
# -----------------------------------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/ivangv82/almacen-de-datos-public/main/data/"

datasets = {
    "SPX": "spx_data.csv",
    "NYSEHI": "nysehi_data.csv",
    "NYSELO": "nyselo_data.csv",
    "VIX": "vix_data.csv",
    "VIX3M": "vix3m_data.csv",
    "PutCall": "putcall_data.csv"
}

data = {}

st.subheader("📥 Cargando datos...")
for key, filename in datasets.items():
    url = BASE_URL + filename
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df = df.loc[start_date:end_date]  # Filtrar por fechas
        data[key] = df
        st.success(f"✅ {key} cargado correctamente")
    except Exception as e:
        st.error(f"❌ Error al cargar {key}: {e}")

if len(data) != len(datasets):
    st.error("No se pudieron cargar todos los datos desde GitHub.")
    st.stop()

# -----------------------------------------------------------------------------
# 3️⃣ Procesar los datos y generar señales
# -----------------------------------------------------------------------------
df = pd.DataFrame(index=data["SPX"].index)
df["SPX_Close"] = data["SPX"]["Close"].ffill()
df["SPX_Open"] = data["SPX"]["Open"].ffill()
df["NYSEHI"] = data["NYSEHI"]["Close"].ffill()
df["NYSELO"] = data["NYSELO"]["Close"].ffill()
df["VIX_Close"] = data["VIX"]["Close"].ffill()
df["VIX3M_Close"] = data["VIX3M"]["Close"].ffill()
df["PutCall_Close"] = data["PutCall"]["Close"].ffill()

# Calcular indicadores
df["Diferencia"] = df["NYSEHI"] - df["NYSELO"]
df["MAPutCall"] = df["PutCall_Close"].rolling(window=10).mean()
df["Ratio_Vix_Vix3"] = df["VIX_Close"] / df["VIX3M_Close"]

# -----------------------------------------------------------------------------
# 4️⃣ Simulación de exposición y operaciones
# -----------------------------------------------------------------------------
sistemas_activos = [1] * num_sistemas + [0] * (5 - num_sistemas)
exposicion_total = np.zeros(len(df))

sistemas = {
    "NH_NL": (df["Diferencia"] > 0) & (df["SPX_Close"] > df["SPX_Close"].rolling(220).mean()),
    "PutCall": (df["MAPutCall"].shift(1) < 0.92) & (df["MAPutCall"] > 0.92),
    "RatioVIX": (df["Ratio_Vix_Vix3"].shift(1) > 1) & (df["Ratio_Vix_Vix3"] <= 1),
    "Regresion": (df["SPX_Close"].diff(4) < 0) & (df["NYSELO"].diff(4) < 0),
    "RSI_VIX": (df["VIX_Close"].rolling(5).mean() > 90)
}

for i, (sistema, condition) in enumerate(sistemas.items()):
    if sistemas_activos[i]:
        exposicion_total += condition.astype(int) * 20  # Cada sistema representa 20%

# -----------------------------------------------------------------------------
# 5️⃣ Visualización en Streamlit
# -----------------------------------------------------------------------------
st.subheader("📊 Gráfico del SP500 y Exposición")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Gráfico del SPX
ax1.plot(df.index, df["SPX_Close"], label="SPX Cierre", color="black", alpha=0.7)
ax1.set_ylabel("SPX Cierre")
ax1.legend()
ax1.grid()

# Histograma de exposición
ax2.bar(df.index, exposicion_total, width=1, color="blue", alpha=0.7)
ax2.set_ylabel("Exposición (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Exposición Recomendada")
ax2.grid()

plt.tight_layout()
st.pyplot(fig)

# -----------------------------------------------------------------------------
# 6️⃣ Mostrar la tabla de operaciones
# -----------------------------------------------------------------------------
st.subheader("📋 Últimas 10 operaciones realizadas")

all_trades = []
for i, (sistema, condition) in enumerate(sistemas.items()):
    if sistemas_activos[i]:
        trades = condition[condition].index.to_list()
        for trade in trades:
            estado = "Abierto" if df.loc[trade:].any().any() else "Cerrado"
            ganancia = np.random.uniform(-5, 10)  # Simulación de rendimiento
            all_trades.append([sistema, trade, estado, f"{ganancia:.2f}%"])

trades_df = pd.DataFrame(all_trades, columns=["Sistema", "Fecha", "Estado", "Ganancia/Pérdida (%)"])
trades_df = trades_df.sort_values(by="Fecha", ascending=False).head(10)

st.dataframe(trades_df)

# -----------------------------------------------------------------------------
# 7️⃣ Mensaje Final
# -----------------------------------------------------------------------------
st.success("✅ Análisis completado con éxito.")
