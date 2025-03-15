import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# Configuración de la App en Streamlit
# =============================================================================
st.title("🔍 Estrategias de Inversión en el SP500")

# Seleccionar cuántos sistemas activar (de 1 a 5)
num_sistemas = st.selectbox("Selecciona cuántos sistemas activar:", [1, 2, 3, 4, 5], index=4)

# Seleccionar rango de fechas
fecha_inicio = st.date_input("Fecha de inicio", datetime(2020, 1, 1))
fecha_fin = st.date_input("Fecha de fin (dejar en blanco para usar hoy)", datetime.today())

# Convertir fechas a formato Pandas
start_date = pd.Timestamp(fecha_inicio)
end_date = pd.Timestamp(fecha_fin) if fecha_fin else pd.Timestamp(datetime.today())

# =============================================================================
# Cargar datos desde GitHub (archivos CSV previamente subidos)
# =============================================================================
BASE_URL = "https://raw.githubusercontent.com/ivangv82/almacen-de-datos/main/data/"

datasets = {
    "SPX": "spx_data.csv",
    "NYSEHI": "nysehi_data.csv",
    "NYSELO": "nyselo_data.csv",
    "VIX": "vix_data.csv",
    "VIX3M": "vix3m_data.csv",
    "PutCall": "putcall_data.csv"
}

data_raw = {}
for key, filename in datasets.items():
    url = BASE_URL + filename
    try:
        data_raw[key] = pd.read_csv(url, index_col=0, parse_dates=True)
        st.write(f"✅ {key} cargado correctamente desde GitHub")
    except Exception as e:
        st.write(f"❌ Error al cargar {key}: {e}")

if not data_raw:
    st.error("No se pudieron cargar los datos desde GitHub.")
    st.stop()
else:
    st.write("✅ Datos cargados desde GitHub")

# =============================================================================
# Crear DataFrame maestro y calcular indicadores
# =============================================================================
master = pd.DataFrame(index=data_raw["SPX"].index)
master["SPX_Close"]    = data_raw["SPX"]["Close"].ffill()
master["SPX_Open"]     = data_raw["SPX"]["Open"].ffill()
master["NYSEHI"]       = data_raw["NYSEHI"]["Close"].ffill()
master["NYSELO"]       = data_raw["NYSELO"]["Close"].ffill()
master["VIX_Close"]    = data_raw["VIX"]["Close"].ffill()
master["VIX3M_Close"]  = data_raw["VIX3M"]["Close"].ffill()
master["PutCall_Close"]= data_raw["PutCall"]["Close"].ffill()

# Indicadores para el Sistema NH_NL
master["Diferencia"] = master["NYSEHI"] - master["NYSELO"]
master["MA_SPX_220"] = master["SPX_Close"].rolling(window=220, min_periods=220).mean()

# Sistema RSI VIX
def built_in_RSI_equivalent(prices, period):
    P = 0.0
    N = 0.0
    rsi = np.full(len(prices), np.nan)
    prices_arr = prices.values
    for i in range(1, len(prices_arr)):
        diff = prices_arr[i] - prices_arr[i - 1]
        W = diff if diff > 0 else 0
        S = -diff if diff < 0 else 0
        P = ((period - 1) * P + W) / period
        N = ((period - 1) * N + S) / period
        if i >= period:
            rsi[i] = 100 * P / (P + N) if (P + N) != 0 else 0
    return pd.Series(rsi, index=prices.index)

master["RSI_VIX"] = built_in_RSI_equivalent(master["VIX_Close"], 5)
master["MA5"]     = master["SPX_Close"].rolling(window=5, min_periods=5).mean()
master["MA200"]   = master["SPX_Close"].rolling(window=200, min_periods=200).mean()

# Sistema PutCall
master["MAPutCall"] = master["PutCall_Close"].rolling(window=10, min_periods=10).mean()

# Sistema Regresion
def linreg_slope_amibroker(series, window):
    slopes = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        y = series[i - (window - 1): i + 1]
        x = np.arange(window)
        if np.any(np.isnan(y)):
            continue
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(x, y)
        slopes[i] = slope * (window - 1)
    return slopes

master["REG_NYLOW"] = linreg_slope_amibroker(master["NYSELO"], 4)
master["REG_SPX"]   = linreg_slope_amibroker(master["SPX_Close"], 4)

# Sistema Ratio VIX-VIX3
master["Ratio_Vix_Vix3"] = master["VIX_Close"] / master["VIX3M_Close"]

# =============================================================================
# Simulación de exposición y operaciones
# Cada sistema aporta 20% de exposición cuando está "in market"
# Se definen 5 sistemas:
#  - NH_NL: (Diferencia > 0) y (SPX_Close > MA_SPX_220)
#  - PutCall: (MAPutCall cruza de abajo hacia arriba el umbral 0.92)
#  - RatioVIX: (Ratio_Vix_Vix3 cruza de arriba hacia abajo 1)
#  - Regresion: (REG_SPX < 0) y (REG_NYLOW < 0)
#  - RSI_VIX: (RSI_VIX > 90)
# =============================================================================
sistemas_activos = [1] * num_sistemas + [0] * (5 - num_sistemas)
exposicion_total = np.zeros(len(master))

# Definir las condiciones para cada sistema:
sistemas = {
    "NH_NL": (master["Diferencia"] > 0) & (master["SPX_Close"] > master["MA_SPX_220"]),
    "PutCall": (master["MAPutCall"].shift(1) < 0.92) & (master["MAPutCall"] > 0.92),
    "RatioVIX": (master["Ratio_Vix_Vix3"].shift(1) > 1) & (master["Ratio_Vix_Vix3"] <= 1),
    "Regresion": (master["REG_SPX"] < 0) & (master["REG_NYLOW"] < 0),
    "RSI_VIX": (master["RSI_VIX"] > 90)
}

for i, (sistema, condition) in enumerate(sistemas.items()):
    if sistemas_activos[i]:
        exposicion_total += condition.astype(int) * 20  # Cada sistema representa 20%

# =============================================================================
# Simulación de operaciones (para fines ilustrativos)
# En este ejemplo, por cada sistema activo se extraen las fechas en las que la condición es True
# y se asigna un % de ganancia aleatorio (simulación dummy).
# =============================================================================
all_trades = []
for i, (sistema, condition) in enumerate(sistemas.items()):
    if sistemas_activos[i]:
        trade_dates = condition[condition].index.to_list()
        for trade in trade_dates:
            gain_pct = np.random.uniform(-5, 10)  # Simulación aleatoria
            # En este ejemplo, todas las operaciones se consideran cerradas
            all_trades.append([sistema, trade, "Cerrado", round(gain_pct, 2)])
trades_df = pd.DataFrame(all_trades, columns=["Sistema", "Fecha", "Estado", "Ganancia/Pérdida (%)"])
trades_df = trades_df.sort_values(by="Fecha", ascending=False).head(10)

# =============================================================================
# Visualización en Streamlit
# =============================================================================
st.subheader("📊 Gráfico del SP500 y Exposición")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Gráfico superior: Precio del SPX
ax1.plot(master.index, master["SPX_Close"], label="SPX Cierre", color="black", alpha=0.7)
ax1.set_ylabel("SPX Cierre")
ax1.legend()
ax1.grid()

# Gráfico inferior: Histograma de exposición (0 a 100%)
ax2.bar(master.index, exposicion_total, width=1, color="blue", alpha=0.7)
ax2.set_ylabel("Exposición (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Exposición Recomendada")
ax2.grid()

plt.tight_layout()
st.pyplot(fig)

st.subheader("📋 Últimas 10 operaciones realizadas")
st.dataframe(trades_df)
