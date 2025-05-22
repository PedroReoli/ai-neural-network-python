# 💹 NeuroTrader v2 – Agente Financeiro Autodidata

Um sistema avançado de trading que combina aprendizado supervisionado e por reforço para operar no mercado financeiro.

## 🎯 Características

- Previsão de tendências usando LSTM
- Agente de trading baseado em PPO (Proximal Policy Optimization)
- Ambiente de simulação personalizado
- Análise técnica integrada
- Visualização de desempenho

## 📋 Requisitos

- Python 3.8+
- TensorFlow 2.8+
- PyTorch
- Gymnasium
- Stable-Baselines3
- YFinance
- Pandas
- NumPy
- Matplotlib

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/NeuroTrader.git
cd NeuroTrader
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Uso

1. Execute o script principal:
```bash
python main.py
```

O sistema irá:
- Baixar dados históricos do Yahoo Finance
- Treinar o modelo de previsão de tendências
- Treinar o agente de trading
- Salvar os modelos treinados
- Mostrar um gráfico do desempenho

## 📊 Componentes

### DataFetcher
- Responsável por baixar e pré-processar dados do mercado
- Calcula indicadores técnicos
- Prepara dados para treinamento

### TrendPredictor
- Modelo LSTM para prever tendências
- Previsão de direção e intensidade
- Normalização de dados

### TradingEnv
- Ambiente de simulação personalizado
- Sistema de recompensas baseado em lucro/prejuízo
- Estado do mercado e portfólio

### TradingAgent
- Agente PPO para tomada de decisões
- Aprendizado por reforço
- Política de trading otimizada

## 📈 Personalização

Você pode personalizar o sistema modificando:
- Símbolo da ação (`symbol` em `DataFetcher`)
- Período de dados (`start_date` e `end_date`)
- Parâmetros de treinamento
- Indicadores técnicos
- Estratégia de recompensa

## ⚠️ Aviso

Este sistema é para fins educacionais e de pesquisa. Não use para trading real sem testes extensivos e compreensão completa dos riscos envolvidos.

## 📝 Licença

MIT License 