import matplotlib.pyplot as plt
from data_fetcher import DataFetcher
from trend_predictor import TrendPredictor
from trading_env import TradingEnv
from trading_agent import TradingAgent, TradingCallback

def plot_portfolio_value(portfolio_values):
    """Plot the portfolio value over time"""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.show()

def main():
    # Initialize components
    print("Initializing NeuroTrader v2...")
    
    # Fetch and prepare data
    print("Fetching market data...")
    data_fetcher = DataFetcher(symbol='AAPL')
    data = data_fetcher.fetch_data()
    
    if data is None:
        print("Error: Could not fetch data. Exiting...")
        return
    
    # Train trend predictor
    print("Training trend predictor...")
    X, y_direction, y_intensity = data_fetcher.get_training_data()
    trend_predictor = TrendPredictor()
    trend_predictor.train(X, y_direction, y_intensity, epochs=50)
    
    # Create and train trading environment
    print("Setting up trading environment...")
    env = TradingEnv(data)
    agent = TradingAgent(env)
    
    # Train the agent
    print("Training trading agent...")
    callback = TradingCallback()
    agent.train(total_timesteps=100000)
    
    # Save models
    print("Saving models...")
    trend_predictor.save_model('trend_predictor.h5')
    agent.save('trading_agent')
    
    # Plot results
    print("Plotting results...")
    plot_portfolio_value(callback.portfolio_values)
    
    print("Training complete! Models saved as 'trend_predictor.h5' and 'trading_agent'")

if __name__ == "__main__":
    main() 