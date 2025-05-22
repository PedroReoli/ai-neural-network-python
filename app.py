import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from data_fetcher import DataFetcher
from trend_predictor import TrendPredictor
from trading_env import TradingEnv
from trading_agent import TradingAgent, TradingCallback
import ta
import pandas_ta as pta

# Configuração da página
st.set_page_config(
    page_title="NeuroTrader v2",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stSelectbox {
        background-color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configurações")
symbol = st.sidebar.text_input("Símbolo da Ação", "AAPL")
start_date = st.sidebar.date_input(
    "Data Inicial",
    datetime.now() - timedelta(days=365)
)
end_date = st.sidebar.date_input(
    "Data Final",
    datetime.now()
)

# Parâmetros de treinamento
st.sidebar.subheader("Parâmetros de Treinamento")
initial_balance = st.sidebar.number_input("Capital Inicial ($)", 10000, 1000000, 10000)
training_epochs = st.sidebar.slider("Épocas de Treinamento", 10, 200, 50)
timesteps = st.sidebar.slider("Timesteps", 10000, 500000, 100000)

# Indicadores técnicos
st.sidebar.subheader("Indicadores Técnicos")
show_rsi = st.sidebar.checkbox("RSI", True)
show_macd = st.sidebar.checkbox("MACD", True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", True)

# Título principal
st.title("💹 NeuroTrader v2 - Plataforma de Trading Inteligente")

# Função para carregar e processar dados
@st.cache_data
def load_data(symbol, start_date, end_date):
    data_fetcher = DataFetcher(symbol=symbol, start_date=start_date, end_date=end_date)
    data = data_fetcher.fetch_data()
    return data

# Função para plotar gráfico de preços
def plot_price_chart(data):
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Preço'
    ))
    
    # Adicionar indicadores técnicos
    if show_bollinger:
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = ta.volatility.bollinger_bands(data['Close'])
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Superior', line=dict(color='rgba(250, 0, 0, 0.3)')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Inferior', line=dict(color='rgba(0, 250, 0, 0.3)')))
    
    fig.update_layout(
        title=f'Gráfico de Preços - {symbol}',
        yaxis_title='Preço ($)',
        xaxis_title='Data',
        template='plotly_dark'
    )
    
    return fig

# Função para plotar indicadores técnicos
def plot_indicators(data):
    fig = go.Figure()
    
    if show_rsi:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='yellow')))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    
    if show_macd:
        macd = ta.trend.macd(data['Close'])
        fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')))
    
    fig.update_layout(
        title='Indicadores Técnicos',
        template='plotly_dark'
    )
    
    return fig

# Função para treinar o modelo
def train_model(data, epochs, timesteps):
    with st.spinner('Treinando modelo...'):
        # Treinar predictor
        X, y_direction, y_intensity = data_fetcher.get_training_data()
        trend_predictor = TrendPredictor()
        trend_predictor.train(X, y_direction, y_intensity, epochs=epochs)
        
        # Treinar agente
        env = TradingEnv(data, initial_balance=initial_balance)
        agent = TradingAgent(env)
        callback = TradingCallback()
        agent.train(total_timesteps=timesteps)
        
        return trend_predictor, agent, callback

# Interface principal
if st.button("Carregar Dados"):
    data = load_data(symbol, start_date, end_date)
    
    if data is not None:
        # Layout em colunas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(plot_price_chart(data), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_indicators(data), use_container_width=True)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Preço Atual", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Variação Diária", f"{data['Returns'].iloc[-1]*100:.2f}%")
        with col3:
            st.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
        with col4:
            st.metric("Volatilidade", f"{data['Volatility'].iloc[-1]*100:.2f}%")
        
        # Treinar modelo
        if st.button("Iniciar Treinamento"):
            trend_predictor, agent, callback = train_model(data, training_epochs, timesteps)
            
            # Plotar resultados do treinamento
            st.subheader("Resultados do Treinamento")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=callback.portfolio_values, name='Valor do Portfólio'))
            fig.update_layout(
                title='Evolução do Portfólio Durante o Treinamento',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Salvar modelos
            trend_predictor.save_model('trend_predictor.h5')
            agent.save('trading_agent')
            st.success("Modelos treinados e salvos com sucesso!")
    else:
        st.error("Erro ao carregar dados. Verifique o símbolo e as datas.")

# Rodapé
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Desenvolvido com ❤️ por NeuroTrader v2</p>
        <p>⚠️ Este sistema é para fins educacionais e de pesquisa.</p>
    </div>
""", unsafe_allow_html=True) 