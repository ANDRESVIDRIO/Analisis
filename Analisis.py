import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime, timedelta
from google import genai


#############################################
token = "AIzaSyB1dzithfUMUBywFvdDywU8mT5XKbB_xS8"
client = genai.client(api_key=token)
#############################################

st.set_page_config(page_title='Advanced Financial Analysis', layout='wide')

st.title('Análisis Financiero Avanzado de Emisoras')

# Input de la emisora
symbol = st.text_input('Ingrese el símbolo de la emisora (por ejemplo, AAPL)', 'AAPL')

# Función para obtener información de la compañía
def get_company_info(ticker):
    try:
        info = ticker.info
        return {
            'Nombre': info.get('shortName', 'Falta de información'),
            'País': info.get('country', 'Falta de información'),
            'Sector': info.get('sector', 'Falta de información'),
            'Industria': info.get('industry', 'Falta de información'),
            'Descripción': info.get('longBusinessSummary', 'Falta de información'),
            'Beta': info.get('beta', 'Falta de información'),
            'Forward PE': info.get('forwardPE', 'Falta de información'),
            'Price to Book': info.get('priceToBook', 'Falta de información'),
            'Market Cap': info.get('marketCap', 'Falta de información'),
            'Dividend Yield': info.get('dividendYield', 'Falta de información')
        }
    except Exception as e:
        st.error(f'Error al obtener la información de la emisora: {e}')
        return {}

# Obtener los datos de la emisora
ticker = yf.Ticker(symbol)
info = get_company_info(ticker)

# Mostrar información general
st.header('Descripción de la Compañía')
for key, value in info.items():
    st.write(f'**{key}**: {value}')

promt = "este es el texto a traducir en 500 caracteres"
response = client.models.generate_content( model="gemini-2.0-flash", contents= promt + info)
print(response)


# Análisis estadístico
st.header('Análisis Estadístico')
try:
    data = ticker.history(period='5y')['Close']
    stats = {
        'Media': data.mean(),
        'Mediana': data.median(),
        'Desviación Estándar': data.std(),
        'Mínimo': data.min(),
        'Máximo': data.max(),
        'Coeficiente de Variación': data.std() / data.mean()
    }
    for key, value in stats.items():
        st.write(f'**{key}**: {value:.2f}')
except Exception as e:
    st.error(f'Error en el análisis estadístico: {e}')

# Gráfico de precios vs índice
st.header('Gráfico de Precios vs Índice')
period = st.selectbox('Periodo', ['1y', '5y', '10y'])
index = st.text_input('Ingrese el índice de referencia (por ejemplo, ^GSPC)', '^GSPC')

try:
    data = ticker.history(period=period)['Close']
    index_data = yf.Ticker(index).history(period=period)['Close']
    data = data / data.iloc[0] * 100
    index_data = index_data / index_data.iloc[0] * 100
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=symbol)
    plt.plot(index_data, label=index)
    plt.title(f'Comparativa de {symbol} vs {index} (Indexado)')
    plt.legend()
    st.pyplot(plt)
except Exception as e:
    st.error(f'Error al cargar el gráfico: {e}')

# Simulación Montecarlo
st.header('Simulación Montecarlo')
days = st.slider('Días a proyectar', 30, 365, 180)
try:
    returns = data.pct_change().dropna()
    last_price = data[-1]
    sim_count = 1000
    sim_df = pd.DataFrame()
    final_prices = []
    for _ in range(sim_count):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
        sim_df[len(sim_df.columns)] = prices
        final_prices.append(prices[-1])
    plt.figure(figsize=(10, 5))
    plt.plot(sim_df)
    plt.title(f'Simulación Montecarlo de {symbol}')
    st.pyplot(plt)
    final_prices = np.array(final_prices)
    scenarios = {
        'más de 10%': np.mean(final_prices >= last_price * 1.10),
        'más de 5%': np.mean(final_prices >= last_price * 1.05),
        'más de 0%': np.mean(final_prices >= last_price),
        'menos de 5%': np.mean(final_prices <= last_price * 0.95),
        'menos de 10%': np.mean(final_prices <= last_price * 0.90)
    }
    st.header('Probabilidades de Escenarios')
    for scenario, prob in scenarios.items():
        st.write(f'**Probabilidad de {scenario}**: {prob * 100:.2f}%')
except Exception as e:
    st.error(f'Error en la simulación Montecarlo: {e}')

st.write('Aplicación creada para el análisis financiero avanzado utilizando Yahoo Finance y Gemini.')
