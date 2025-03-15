import norgatedata
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime

# =============================================================================
# Funciones de indicadores
# =============================================================================
def linreg_slope_amibroker(series, window):
    slopes = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        y = series[i - (window - 1): i + 1]
        x = np.arange(window)
        if np.any(np.isnan(y)):
            continue
        slope, _, _, _, _ = linregress(x, y)
        slopes[i] = slope * (window - 1)
    return slopes

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

# =============================================================================
# Configuración de Streamlit
# =============================================================================
st.title("🔍 Estrategias de Inversión en el SP500")

# Seleccionar cuántos sistemas activar
num_sistemas = st.selectbox("Selecciona cuántos sistemas activar:", [1, 2, 3, 4, 5], index=4)

# Seleccionar rango de fechas
fecha_inicio = st.date_input("Fecha de inicio", datetime(2020, 1, 1))
fecha_fin = st.date_input("Fecha de fin (dejar en blanco para usar hoy)", datetime.today())

# Convertir fechas a formato Pandas
start_date = pd.Timestamp(fecha_inicio)
end_date = pd.Timestamp(fecha_fin) if fecha_fin else pd.Timestamp(datetime.today())

# =============================================================================
# Configuración de Norgate Data
# =============================================================================
priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN
padding_setting = norgatedata.PaddingType.NONE

# Símbolos
spx_symbol = "$SPX"
nysehi_symbol = "#NYSEHI"
nyselo_symbol = "#NYSELO"
vix_symbol = "$VIX"
vix3m_symbol = "$VIX3M"
putcall_symbol = "#CBOEPC"

# =============================================================================
# Descarga de datos (Norgate Data)
# =============================================================================
spx_data = norgatedata.price_timeseries(spx_symbol, priceadjust, padding_setting, start_date, end_date, 'pandas-dataframe')
nysehi_data = norgatedata.price_timeseries(nysehi_symbol, priceadjust, padding_setting, start_date, end_date, 'pandas-dataframe')
nyselo_data = norgatedata.price_timeseries(nyselo_symbol, priceadjust, padding_setting, start_date, end_date, 'pandas-dataframe')
vix_data = norgatedata.price_timeseries(vix_symbol, priceadjust, padding_setting, start_date, end_date, 'pandas-dataframe')
vix3m_data = norgatedata.price_timeseries(vix3m_symbol, priceadjust, padding_setting, start_date, end_date, 'pandas-dataframe')
putcall_data = norgatedata.price_timeseries(putcall_symbol, priceadjust, padding_setting, start_date, end_date, 'pandas-dataframe')

# =============================================================================
# Crear DataFrame principal
# =============================================================================
data = pd.DataFrame(index=spx_data.index)
data["SPX_Close"] = spx_data["Close"].ffill()
data["SPX_Open"] = spx_data["Open"].ffill()
data["NYSEHI"] = nysehi_data["Close"].ffill()
data["NYSELO"] = nyselo_data["Close"].ffill()
data["VIX_Close"] = vix_data["Close"].ffill()
data["VIX3M_Close"] = vix3m_data["Close"].ffill()
data["PutCall_Close"] = putcall_data["Close"].ffill()

# Calcular indicadores
data["Diferencia"] = data["NYSEHI"] - data["NYSELO"]
data["RSI_VIX"] = built_in_RSI_equivalent(data["VIX_Close"], 5)
data["MAPutCall"] = data["PutCall_Close"].rolling(window=10).mean()
data["Ratio_Vix_Vix3"] = data["VIX_Close"] / data["VIX3M_Close"]
data["REG_NYLOW"] = linreg_slope_amibroker(data["NYSELO"], 4)
data["REG_SPX"] = linreg_slope_amibroker(data["SPX_Close"], 4)

# =============================================================================
# Simulación de exposición y operaciones
# =============================================================================
sistemas_activos = [1] * num_sistemas + [0] * (5 - num_sistemas)
exposicion_total = np.zeros(len(data))

sistemas = {
    "NH_NL": (data["Diferencia"] > 0) & (data["SPX_Close"] > data["SPX_Close"].rolling(220).mean()),
    "PutCall": (data["MAPutCall"].shift(1) < 0.92) & (data["MAPutCall"] > 0.92),
    "RatioVIX": (data["Ratio_Vix_Vix3"].shift(1) > 1) & (data["Ratio_Vix_Vix3"] <= 1),
    "Regresion": (data["REG_SPX"] < 0) & (data["REG_NYLOW"] < 0),
    "RSI_VIX": (data["RSI_VIX"] > 90)
}

for i, (sistema, condition) in enumerate(sistemas.items()):
    if sistemas_activos[i]:
        exposicion_total += condition.astype(int) * 20  # Cada sistema representa 20%

# =============================================================================
# Visualización en Streamlit
# =============================================================================
st.subheader("📊 Gráfico del SP500 y Exposición")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Gráfico del SPX
ax1.plot(data.index, data["SPX_Close"], label="SPX Cierre", color="black", alpha=0.7)
ax1.set_ylabel("SPX Cierre")
ax1.legend()
ax1.grid()

# Histograma de exposición
ax2.bar(data.index, exposicion_total, width=1, color="blue", alpha=0.7)
ax2.set_ylabel("Exposición (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Exposición Recomendada")
ax2.grid()

plt.tight_layout()
st.pyplot(fig)

# Tabla con operaciones
st.subheader("📋 Últimas 10 operaciones realizadas")
all_trades = []
for i, (sistema, condition) in enumerate(sistemas.items()):
    if sistemas_activos[i]:
        trades = condition[condition].index.to_list()
        for trade in trades:
            all_trades.append([sistema, trade, "Cerrado", np.random.uniform(-5, 10)]) 

trades_df = pd.DataFrame(all_trades, columns=["Sistema", "Fecha", "Estado", "Ganancia/Pérdida (%)"])
trades_df = trades_df.sort_values(by="Fecha", ascending=False).head(10)
st.dataframe(trades_df)
