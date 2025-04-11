import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import norm

# Configuración del estilo CSS personalizado
st.image("LOGO.png",width=300)
#####################################################################################
#st.set_page_config(page_title='Advanced Financial Analysis')

st.title('Advanced Financial Analysis')

seccion = st.sidebar.radio("Herramientas", ["Informacion general","Análisis Estadístico","Comparactiva contra el indice","Monte Carlo","Medias móviles","Cartera Eficiente"])




#############################################
#token = "AIzaSyB1dzithfUMUBywFvdDywU8mT5XKbB_xS8"
#client = genai.client(api_key=token)
#############################################


# Input de la emisora
symbol = st.text_input('Ingrese el símbolo de la emisora (por ejemplo, AAPL)', 'AAPL')

# Sección: Información General
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

if seccion == "Informacion general":
    st.header("**Información general**")

    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            info = get_company_info(ticker)
            for key, value in info.items():
                st.write(f'**{key}**: {value}')
        except Exception as e:
            st.error(f"No se pudo obtener información para el símbolo '{symbol}': {e}")
    else:
        st.warning("Por favor, ingresa un símbolo válido.")




# Obtener los datos de la emisora
ticker = yf.Ticker(symbol)
info = get_company_info(ticker)

# Mostrar información general
#st.header('Descripción de la Compañía')
#for key, value in info.items():
#    st.write(f'**{key}**: {value}')

#promt = "este es el texto a traducir en 500 caracteres"
#response = client.models.generate_content( model="gemini-2.0-flash", contents= promt + info)
#print(response)



if seccion == "Análisis Estadístico":
    st.header("**Análisis Estadístico**")

    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5y')['Close'].dropna()

            # Cálculo de rendimientos logarítmicos
            returns = np.log(data / data.shift(1)).dropna()

            # Media y desviación estándar para la curva normal
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 100)
            y = norm.pdf(x, mu, sigma)

            # Gráfico con Plotly
            fig = go.Figure()

            # Histograma de rendimientos
            fig.add_trace(go.Histogram(
                x=returns,
                histnorm='probability density',
                name='Distribución de rendimientos',
                marker_color='lightblue',
                opacity=0.75,
                nbinsx=50
            ))

            # Curva de distribución normal teórica
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Distribución Normal',
                line=dict(color='red', width=2)
            ))

            fig.update_layout(
                title=f'Distribución de Rendimientos - {symbol.upper()}',
                xaxis_title='Rendimientos logarítmicos',
                yaxis_title='Densidad',
                template='plotly_white',
                showlegend=True,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Estadísticas adicionales
            st.subheader("**Estadísticas de los Rendimientos**")
            st.write(f"**Media**: {mu:.4f}")
            st.write(f"**Desviación Estándar**: {sigma:.4f}")
            st.write(f"**Asimetría**: {returns.skew():.4f}")
            st.write(f"**Curtosis**: {returns.kurtosis():.4f}")

        except Exception as e:
            st.error(f'Error en el análisis estadístico: {e}')
    else:
        st.warning("Por favor, ingresa un símbolo válido.")


if seccion == "Comparactiva contra el indice":
   st.header("**Comparactiva contra el indice**") 
   # Gráfico de precios vs índice
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

if seccion == "Informacion general":
   st.header("**Informacion general**") 
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


if seccion == "Medias móviles":
 
   st.title("📈 Análisis Técnico de Acciones")
   st.write("Este dashboard permite realizar un análisis técnico detallado con indicadores clave.")

   # Entrada del usuario
   ticker = st.sidebar.text_input("Ingrese el Ticker de la Acción (Ej: AAPL)", value="AAPL")
   periodo = st.sidebar.selectbox("Selecciona el Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=4)
   intervalo = st.sidebar.selectbox("Intervalo de Datos", ["1d", "1h", "30m", "15m", "5m"], index=0)

   # Descargar datos
   @st.cache_data
   def obtener_datos(ticker, periodo, intervalo):
       return yf.download(ticker, period=periodo, interval=intervalo)

   df = obtener_datos(ticker, periodo, intervalo)

   # Verificar si hay datos
   if df.empty:
       st.error("No se encontraron datos para el ticker seleccionado. Intente otro.")
       st.stop()

   # Calcular indicadores técnicos
   df["SMA_50"] = df["Close"].rolling(window=50).mean()  # Media Móvil Simple 50
   df["SMA_1"] = df["Close"].rolling(window=1).mean()
   df["SMA_200"] = df["Close"].rolling(window=200).mean()  # Media Móvil Simple 200
   df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()  # Media Móvil Exponencial 20
 
   # Bandas de Bollinger
   df["Upper_BB"] = df["SMA_50"] + (df["Close"].rolling(window=50).std().iloc[:, 0] * 2)
   df["Lower_BB"] = df["SMA_50"] - (df["Close"].rolling(window=50).std().iloc[:, 0] * 2)

 
   # Índice de Fuerza Relativa (RSI)
   def calcular_rsi(series, periodo=14):
       delta = series.diff()
       ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
       perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
       rs = ganancia / perdida
       return 100 - (100 / (1 + rs))

   df["RSI"] = calcular_rsi(df["Close"])

   # MACD
   df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
   df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()



   # 📊 Gráfico de Precios con Medias Móviles y Bandas de Bollinger
   fig = go.Figure()

   fig.add_trace(go.Scatter(x=df.index, y=df["SMA_1"], mode="lines", name="Precio", line=dict(color="yellow", width=1)))
   fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50", line=dict(color="blue", width=1)))
   fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], mode="lines", name="SMA 200", line=dict(color="red", width=1)))
   fig.add_trace(go.Scatter(x=df.index, y=df["Upper_BB"], mode="lines", name="Banda Superior", line=dict(color="green", width=1, dash="dot")))
   fig.add_trace(go.Scatter(x=df.index, y=df["Lower_BB"], mode="lines", name="Banda Inferior", line=dict(color="green", width=1, dash="dot")))

   fig.update_layout(title=f"Análisis Técnico de {ticker}", xaxis_title="Fecha", yaxis_title="Precio", xaxis_rangeslider_visible=False)
   st.plotly_chart(fig, use_container_width=True)

   # 📉 RSI y MACD
   col1, col2 = st.columns(2)

   with col1:
       st.subheader("RSI (Índice de Fuerza Relativa)")
       fig_rsi = go.Figure()
       fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="purple")))
       fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"))
       fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"))
       fig_rsi.update_layout(title="RSI", xaxis_title="Fecha", yaxis_title="Valor", xaxis_rangeslider_visible=False)
       st.plotly_chart(fig_rsi, use_container_width=True)

   with col2:
       st.subheader("MACD (Moving Average Convergence Divergence)")
       fig_macd = go.Figure()
       fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="blue")))
       fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode="lines", name="Señal", line=dict(color="red")))
       fig_macd.update_layout(title="MACD", xaxis_title="Fecha", yaxis_title="Valor", xaxis_rangeslider_visible=False)
       st.plotly_chart(fig_macd, use_container_width=True)


 
   # 📌 Conclusión
   st.markdown("### 📊 **Conclusiones**")
   st.markdown("""
   - **RSI > 70:** La acción podría estar sobrecomprada (posible corrección).  
   - **RSI < 30:** La acción podría estar sobrevendida (posible rebote).  
   - **MACD cruzando por encima de la señal:** Indica tendencia alcista.  
   - **MACD cruzando por debajo de la señal:** Indica tendencia bajista.  
   - **El precio cerca de la Banda Superior de Bollinger:** Posible sobrecompra.  
   - **El precio cerca de la Banda Inferior de Bollinger:** Posible sobreventa.  
   """)


if seccion == "Cartera Eficiente":

    st.title("📊 Frontera Eficiente (2 Activos con Matplotlib)")

    # Inputs
    ticker1 = st.sidebar.text_input("Ticker Activo 1", value="AAPL")
    ticker2 = st.sidebar.text_input("Ticker Activo 2", value="MSFT")
    rf = st.sidebar.number_input("Tasa libre de riesgo (%)", value=3.0) / 100

    # Función para obtener retornos
    def get_returns(ticker):
        data = yf.download(ticker, period="5y")["Adj Close"]
        return data.pct_change().dropna()

    r1 = get_returns(ticker1)
    r2 = get_returns(ticker2)

    # Medias, desviaciones y covarianza
    mu1, std1 = r1.mean(), r1.std()
    mu2, std2 = r2.mean(), r2.std()
    cov = np.cov(r1, r2)[0, 1]

    # Simulación
    pesos = np.linspace(0, 1, 100)
    rend = pesos * mu1 + (1 - pesos) * mu2
    riesgo = np.sqrt(pesos**2 * std1**2 + (1 - pesos)**2 * std2**2 + 2 * pesos * (1 - pesos) * cov)
    sharpe = (rend - rf) / riesgo

    # Índices de portafolios clave
    idx_min = np.argmin(riesgo)
    idx_max = np.argmax(sharpe)

    # Gráfico con matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(riesgo, rend, color="blue", label="Frontera Eficiente")
    ax.scatter(riesgo[idx_min], rend[idx_min], color="red", s=100, label="Menor Riesgo")
    ax.scatter(riesgo[idx_max], rend[idx_max], color="green", s=100, label="Máx Sharpe")
    ax.set_title("Frontera Eficiente de 2 Activos")
    ax.set_xlabel("Riesgo (Desviación Estándar)")
    ax.set_ylabel("Rentabilidad Esperada")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Resultados
    st.subheader("Resultados")
    st.write(f"📉 Menor Riesgo: {pesos[idx_min]*100:.2f}% en {ticker1} / {(1-pesos[idx_min])*100:.2f}% en {ticker2}")
    st.write(f"↪️ Rentabilidad: {rend[idx_min]*100:.2f}%, Riesgo: {riesgo[idx_min]*100:.2f}%")
    st.write("---")
    st.write(f"📈 Máximo Sharpe: {pesos[idx_max]*100:.2f}% en {ticker1} / {(1-pesos[idx_max])*100:.2f}% en {ticker2}")
    st.write(f"↪️ Rentabilidad: {rend[idx_max]*100:.2f}%, Riesgo: {riesgo[idx_max]*100:.2f}%")
    st.write(f"⭐ Sharpe Ratio: {sharpe[idx_max]:.2f}")