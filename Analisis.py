import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup


# Configuración del estilo CSS personalizado
st.image("LOGO.png",width=300)
#####################################################################################
#st.set_page_config(page_title='Advanced Financial Analysis')

st.title('Advanced Financial Analysis')

seccion = st.sidebar.radio("Herramientas", ["Informacion general","Monte Carlo","Medias móviles","Cartera Eficiente","creator of sector ETFs"
""])




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
            
            # Información de la compañía
            info = get_company_info(ticker)
            for key, value in info.items():
                st.write(f'**{key}**: {value}')

            # Precios históricos y rendimientos
            data = ticker.history(period='5y')['Close'].dropna()

            rend_diario = data.pct_change().dropna()
            rend_acumulado = (1 + rend_diario).cumprod() - 1
            rend_anual = rend_diario.mean() * 252

            # Gráfico en plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data.values,
                mode='lines',
                name=f'{symbol}',
                line=dict(color='royalblue')
            ))
            fig.update_layout(
                title=f'Precio Ajustado de {symbol} - Últimos 5 años',
                xaxis_title='Fecha',
                yaxis_title='Precio (USD)',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mostrar rendimientos
            st.subheader("📊 Rendimientos")
            st.write(f"**Rendimiento acumulado (5 años):** {rend_acumulado[-1] * 100:.2f}%")
            st.write(f"**Rendimiento promedio anual:** {rend_anual * 100:.2f}%")

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

if seccion == "Informacion general":
    st.header("**Comparativa contra el índice**") 

    # Parámetros de usuario
    period = st.selectbox('Periodo', ['1y', '5y', '10y'])
    index = st.text_input('Ingrese el índice de referencia (por ejemplo, ^GSPC)', '^GSPC')

    try:
        # Datos del activo y del índice
        precios = ticker.history(period=period)['Close']
        precios_indice = yf.Ticker(index).history(period=period)['Close']

        # Indexar ambos a 100
        precios = precios / precios.iloc[0] * 100
        precios_indice = precios_indice / precios_indice.iloc[0] * 100

        # Gráfico interactivo con plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=precios.index, y=precios,
            mode='lines', name=symbol,
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=precios_indice.index, y=precios_indice,
            mode='lines', name=index,
            line=dict(color='orange')
        ))

        fig.update_layout(
            title=f'Comparativa de {symbol} vs {index} (Indexado a 100)',
            xaxis_title='Fecha',
            yaxis_title='Precio Indexado',
            legend=dict(x=0.01, y=0.99),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f'Error al cargar el gráfico: {e}')

if seccion == "Informacion general":


  st.header("**Análisis Estadístico**")

  try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='5y')['Close'].dropna()

        # Cálculo de rendimientos porcentuales
        returns = data.pct_change().dropna() * 100  # multiplicar por 100 para expresarlo en %

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
            name='Distribución de rendimientos (%)',
            marker_color='lightblue',
            opacity=0.75,
            nbinsx=300
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
            title=f'Distribución de Rendimientos (%) - {symbol.upper()}',
            xaxis_title='Rendimientos diarios (%)',
            yaxis_title='Densidad',
            template='plotly_white',
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Estadísticas adicionales
        st.subheader("**Estadísticas de los Rendimientos (%)**")
        st.write(f"**Media**: {mu:.4f}%")
        st.write(f"**Desviación Estándar**: {sigma:.4f}%")
        st.write(f"**Asimetría**: {returns.skew():.4f}")
        st.write(f"**Curtosis**: {returns.kurtosis():.4f}")

  except Exception as e:
        st.error(f'Error en el análisis estadístico: {e}')
else:
    st.warning("Por favor, ingresa un símbolo válido.")



if seccion == "Monte Carlo":
    st.header("📊 Simulación Monte Carlo")


    days = st.slider("Días a proyectar", 30, 365, 180)

    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5y")["Close"]
        returns = data.pct_change().dropna()
        last_price = data[-1]
        sim_count = 1000
        sim_df = pd.DataFrame()
        final_prices = []

        for i in range(sim_count):
            prices = [last_price]
            for _ in range(days):
                prices.append(prices[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
            sim_df[i] = prices
            final_prices.append(prices[-1])

        # Gráfico con Plotly
        fig = go.Figure()

        for i in range(min(100, sim_count)):
            fig.add_trace(go.Scatter(
                y=sim_df[i],
                mode='lines',
                line=dict(width=1),
                showlegend=False
            ))

        fig.update_layout(
            title=f'Simulación Monte Carlo de {symbol.upper()} ({sim_count} simulaciones)',
            xaxis_title='Días',
            yaxis_title='Precio Simulado',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Probabilidades de escenarios
        final_prices = np.array(final_prices)
        scenarios = {
            'más de 10%': np.mean(final_prices >= last_price * 1.10),
            'más de 5%': np.mean(final_prices >= last_price * 1.05),
            'más de 0%': np.mean(final_prices >= last_price),
            'menos de 5%': np.mean(final_prices <= last_price * 0.95),
            'menos de 10%': np.mean(final_prices <= last_price * 0.90)
        }

        st.subheader("📈 Probabilidades de Escenarios")
        for escenario, probabilidad in scenarios.items():
            st.write(f"**Probabilidad de {escenario}**: {probabilidad * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error en la simulación Montecarlo: {e}")


if seccion == "Medias móviles":
 
   st.title("📈 Análisis Técnico de Acciones")
   st.write("Este dashboard permite realizar un análisis técnico detallado con indicadores clave.")

   # Entrada del usuario
   ticker = symbol
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

   st.title("Optimización de Cartera Eficiente")

   # Ingreso libre de símbolos
   tickers_input = st.text_input("Ingresa los símbolos de acciones o ETFs separados por comas", "AAPL,MSFT,GOOGL")
   tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

   def obtener_pares_finviz(ticker):
      

      headers = {"User-Agent": ""}

      url = f"https://finviz.com/quote.ashx?t={ticker}"
      response = requests.get(url, headers=headers)
      response.raise_for_status()
      soup = BeautifulSoup(response.text, "html.parser")

      link = soup.find("a", string="Peers")
      if link and "href" in link.attrs:
         href = link["href"]
         if "t=" in href:
            tickers_str = href.split("t=")[1]
            pares = tickers_str.split(",")
            return pares
      return []

   # Selección del número de años
   anios = st.slider("Selecciona el número de años de datos históricos", 1, 10, 5)
   fecha_fin = datetime.today()
   fecha_inicio = fecha_fin - timedelta(days=anios * 365)
   st.markdown(f"**Período de análisis:** {fecha_inicio.date()} a {fecha_fin.date()}")

   # Descargar precios históricos usando 'Close'
   @st.cache_data
   def descargar_precios(tickers, start, end):
      datos = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)

      if datos.empty:
         st.warning("Fallo al descargar con fechas, intentando con 'period=5y'...")
         datos = yf.download(tickers, period="5y", group_by='ticker', auto_adjust=False)

      if datos.empty:
         st.error("No se pudieron descargar datos. Verifica los símbolos ingresados.")
         st.stop()

      precios = pd.DataFrame()
      if isinstance(datos.columns, pd.MultiIndex):
         for t in tickers:
            try:
               precios[t] = datos[t]['Close']
            except KeyError:
               st.warning(f"No se encontraron datos para {t}, se omitirá.")
      else:
         if 'Close' in datos.columns:
            precios = datos[['Close']]
            precios.columns = tickers[:1]
         else:
            st.error("No se encontró la columna 'Close'.")
            st.stop()

      precios.dropna(axis=1, inplace=True)
      if precios.empty:
         st.error("No hay suficientes datos válidos tras eliminar columnas con NA.")
         st.stop()

      return precios

   precios = descargar_precios(tickers, fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d"))
   rendimientos = precios.pct_change().dropna()

   # Estadísticas básicas
   retornos_esperados = rendimientos.mean() * 252
   cov = rendimientos.cov() * 252

   # Función de rendimiento y riesgo
   def calcular_portafolio(w, mu, sigma):
      retorno = np.dot(w, mu)
      riesgo = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
      return retorno, riesgo

   # Simulación de portafolios
   def simular_portafolios(mu, sigma, n=5000):
      resultados = {'retorno': [], 'riesgo': [], 'sharpe': [], 'pesos': []}
      for _ in range(n):
         w = np.random.dirichlet(np.ones(len(mu)))
         r, s = calcular_portafolio(w, mu, sigma)
         resultados['retorno'].append(r)
         resultados['riesgo'].append(s)
         resultados['sharpe'].append(r / s if s != 0 else 0)
         resultados['pesos'].append(w)
      return pd.DataFrame(resultados)

   portafolios = simular_portafolios(retornos_esperados.values, cov.values)

   # Portafolio Máxima Sharpe
   def portafolio_max_sharpe(mu, sigma):
      n = len(mu)
      def objetivo(w): return -np.dot(w, mu) / np.sqrt(np.dot(w.T, np.dot(sigma, w)))
      cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
      bounds = tuple((0, 1) for _ in range(n))
      res = minimize(objetivo, np.repeat(1/n, n), bounds=bounds, constraints=cons)
      return res.x

   # Portafolio Mínima Varianza
   def portafolio_min_varianza(sigma):
      n = len(sigma)
      def objetivo(w): return np.dot(w.T, np.dot(sigma, w))
      cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
      bounds = tuple((0, 1) for _ in range(n))
      res = minimize(objetivo, np.repeat(1/n, n), bounds=bounds, constraints=cons)
      return res.x

   w_sharpe = portafolio_max_sharpe(retornos_esperados.values, cov.values)
   ret_sharpe, risk_sharpe = calcular_portafolio(w_sharpe, retornos_esperados.values, cov.values)

   w_minvar = portafolio_min_varianza(cov.values)
   ret_minvar, risk_minvar = calcular_portafolio(w_minvar, retornos_esperados.values, cov.values)

   # Mostrar resultados lado a lado
   col1, col2 = st.columns(2)

   with col1:
      st.subheader("Portafolio Máxima Sharpe Ratio")
      df_sharpe = pd.DataFrame({'Ticker': precios.columns, 'Peso': w_sharpe})
      st.dataframe(df_sharpe.style.format({'Peso': '{:.2%}'}))

   with col2:
      st.subheader("Portafolio Mínima Varianza")
      df_minvar = pd.DataFrame({'Ticker': precios.columns, 'Peso': w_minvar})
      st.dataframe(df_minvar.style.format({'Peso': '{:.2%}'}))

   # Gráfico
   fig, ax = plt.subplots(figsize=(10, 6))
   sc = ax.scatter(portafolios['riesgo'], portafolios['retorno'], 
                   c=portafolios['sharpe'], cmap='viridis', alpha=0.5)
   ax.scatter(risk_sharpe, ret_sharpe, c='red', s=100, label='Máx Sharpe')
   ax.scatter(risk_minvar, ret_minvar, c='blue', s=100, label='Mín Varianza')
   ax.set_xlabel('Riesgo (Desviación estándar)')
   ax.set_ylabel('Retorno Esperado')
   ax.set_title('Frontera de Cartera Eficiente')
   ax.legend()
   plt.colorbar(sc, label='Sharpe Ratio')
   st.pyplot(fig)


if seccion == "creator of sector ETFs":


   st.title("Optimización de Cartera Eficiente")

   def obtener_pares_finviz(ticker):
      try:
         url = f"https://finviz.com/quote.ashx?t={ticker}"
         headers = {"User-Agent": "fc4d1056-21d9-42b0-9dd9-4c947e694cfe"}
         response = requests.get(url, headers=headers, timeout=10)
         response.raise_for_status()

         soup = BeautifulSoup(response.text, "html.parser")
         link = soup.find("a", string="Peers")
         if link and "href" in link.attrs:
            href = link["href"]
            if "t=" in href:
               tickers_str = href.split("t=")[1]
               pares = tickers_str.split(",")
               return list(set(pares))  # elimina repetidos
         st.warning("No se encontraron pares en la página de Finviz.")
      except requests.exceptions.HTTPError as e:
         st.error(f"Error al obtener la información de la emisora: {e}")
      except Exception as e:
         st.error(f"Ocurrió un error inesperado: {e}")
      return []

   ticker_base = symbol #st.text_input("Ticker base para obtener pares desde Finviz", "AAPL")
   tickers = obtener_pares_finviz(ticker_base.upper())

   if not tickers:
      st.warning("No se pudieron obtener pares para el ticker ingresado.")
      st.stop()

   st.write("### Pares detectados:", ", ".join(tickers))

   # Selección del número de años
   anios = st.slider("Selecciona el número de años de datos históricos", 1, 10, 5)
   fecha_fin = datetime.today()
   fecha_inicio = fecha_fin - timedelta(days=anios * 365)
   st.markdown(f"**Período de análisis:** {fecha_inicio.date()} a {fecha_fin.date()}")

   @st.cache_data
   def descargar_precios(tickers, start, end):
      datos = yf.download(tickers, start=start, end=end, group_by="ticker", progress=False)
      precios = pd.DataFrame()
      for t in tickers:
         try:
            precios[t] = datos[t]['Close']
         except Exception:
            st.warning(f"No se pudo obtener precios para {t}.")
      return precios.dropna(axis=1, how='any')

   precios = descargar_precios(tickers, fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d"))
   if precios.empty:
      st.error("No se pudieron descargar precios para los tickers seleccionados.")
      st.stop()

   rendimientos = precios.pct_change().dropna()
   retornos_esperados = rendimientos.mean() * 252
   cov = rendimientos.cov() * 252

   def calcular_portafolio(w, mu, sigma):
      retorno = np.dot(w, mu)
      riesgo = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
      return retorno, riesgo

   def simular_portafolios(mu, sigma, n=5000):
      resultados = {'retorno': [], 'riesgo': [], 'sharpe': [], 'pesos': []}
      for _ in range(n):
         w = np.random.dirichlet(np.ones(len(mu)))
         r, s = calcular_portafolio(w, mu, sigma)
         resultados['retorno'].append(r)
         resultados['riesgo'].append(s)
         resultados['sharpe'].append(r / s)
         resultados['pesos'].append(w)
      return pd.DataFrame(resultados)

   portafolios = simular_portafolios(retornos_esperados.values, cov.values)

   def portafolio_max_sharpe(mu, sigma):
      n = len(mu)
      def objetivo(w): return -np.dot(w, mu) / np.sqrt(np.dot(w.T, np.dot(sigma, w)))
      cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
      bounds = tuple((0, 1) for _ in range(n))
      res = minimize(objetivo, np.repeat(1/n, n), bounds=bounds, constraints=cons)
      return res.x

   def portafolio_min_varianza(mu, sigma):
      n = len(mu)
      def objetivo(w): return np.dot(w.T, np.dot(sigma, w))
      cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
      bounds = tuple((0, 1) for _ in range(n))
      res = minimize(objetivo, np.repeat(1/n, n), bounds=bounds, constraints=cons)
      return res.x

   w_opt = portafolio_max_sharpe(retornos_esperados.values, cov.values)
   w_min = portafolio_min_varianza(retornos_esperados.values, cov.values)
   ret_opt, risk_opt = calcular_portafolio(w_opt, retornos_esperados.values, cov.values)
   ret_min, risk_min = calcular_portafolio(w_min, retornos_esperados.values, cov.values)

   st.subheader("Portafolios Óptimos")
   col1, col2 = st.columns(2)

   with col1:
      st.write("**Máxima Sharpe Ratio**")
      df_opt = pd.DataFrame({'Ticker': precios.columns, 'Peso óptimo': w_opt})
      st.dataframe(df_opt.style.format({'Peso óptimo': '{:.2%}'}))

   with col2:
      st.write("**Mínima Varianza**")
      df_min = pd.DataFrame({'Ticker': precios.columns, 'Peso óptimo': w_min})
      st.dataframe(df_min.style.format({'Peso óptimo': '{:.2%}'}))

   fig, ax = plt.subplots(figsize=(10, 6))
   sc = ax.scatter(portafolios['riesgo'], portafolios['retorno'], 
                   c=portafolios['sharpe'], cmap='viridis', alpha=0.5)
   ax.scatter(risk_opt, ret_opt, c='red', s=100, label='Máx Sharpe')
   ax.scatter(risk_min, ret_min, c='blue', s=100, label='Min Varianza')
   ax.set_xlabel('Riesgo (Desviación estándar)')
   ax.set_ylabel('Retorno Esperado')
   ax.set_title('Frontera de Cartera Eficiente')
   ax.legend()
   plt.colorbar(sc, label='Sharpe Ratio')
   st.pyplot(fig)


if seccion == "Informacion general":


  HEADERS = {"User-Agent": "fc4d1056-21d9-42b0-9dd9-4c947e694cfe"}

  def obtener_pares_finviz(ticker):
    try:
      url = f"https://finviz.com/quote.ashx?t={ticker}"
      response = requests.get(url, headers=HEADERS, timeout=30)
      response.raise_for_status()

      soup = BeautifulSoup(response.text, "html.parser")
      link = soup.find("a", string="Peers")
      if link and "href" in link.attrs:
        href = link["href"]
        if "t=" in href:
          tickers_str = href.split("t=")[1]
          pares = tickers_str.split(",")
          return list(set(pares))  # elimina repetidos
      st.warning("No se encontraron pares en la página de Finviz.")
    except requests.exceptions.HTTPError as e:
      st.error(f"Error al obtener la información de la emisora: {e}")
    except Exception as e:
      st.error(f"Ocurrió un error inesperado: {e}")
    return []

  @st.cache_data(ttl=3600)
  def obtener_datos_fundamentales(tickers):
    datos = []
    for t in tickers:
      try:
        info = yf.Ticker(t).info
        datos.append({
          "Ticker": t,
          "Market Cap": info.get("marketCap"),
          "Forward PE": info.get("forwardPE"),
          "Price/Sales": info.get("priceToSalesTrailing12Months"),
          "Price/Book": info.get("priceToBook"),
          "EV/Revenue": info.get("enterpriseToRevenue"),
          "EV/EBITDA": info.get("enterpriseToEbitda"),
          "Dividend Yield": info.get("dividendYield"),
          "Beta": info.get("beta"),
          "ROE": info.get("returnOnEquity"),
          "Margen Neto": info.get("profitMargins"),
        })
      except Exception as e:
        st.warning(f"No se pudo obtener datos de {t}: {e}")
    return pd.DataFrame(datos)

  st.title("Comparativo")

  ticker = symbol

  if ticker:
    pares = obtener_pares_finviz(ticker)
    if not pares:
      st.error("No se pudieron obtener pares para este ticker.")
    else:
      pares = [t for t in pares if t != ticker]

      st.info(f"Pares de {ticker}: {', '.join(pares)}")

      if len(pares) == 0:
        st.warning("No hay pares disponibles tras excluir el ticker principal.")
      else:
        df = obtener_datos_fundamentales(pares)
        if df.empty:
          st.error("No se pudieron obtener datos fundamentales para los pares.")
        else:
          st.subheader("Datos fundamentales de los pares")
          st.dataframe(df)


