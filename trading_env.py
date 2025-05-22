import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_fee=0.001, slippage=0.001):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.current_step = 0
        
        # Define action space (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space with more features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,),  # Extended feature set
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.total_value = self.balance
        self.returns = []
        self.trades = []
        self.portfolio_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state observation with extended features"""
        current_price = self.data['Close'].iloc[self.current_step]
        current_returns = self.data['Returns'].iloc[self.current_step]
        current_ma5 = self.data['MA5'].iloc[self.current_step]
        current_ma20 = self.data['MA20'].iloc[self.current_step]
        current_rsi = self.data['RSI'].iloc[self.current_step]
        current_macd = self.data['MACD'].iloc[self.current_step]
        current_volume = self.data['Volume'].iloc[self.current_step]
        current_volatility = self.data['Volatility'].iloc[self.current_step]
        current_atr = self.data['ATR'].iloc[self.current_step]
        current_obv = self.data['OBV'].iloc[self.current_step]
        current_vwap = self.data['VWAP'].iloc[self.current_step]
        current_adx = self.data['ADX'].iloc[self.current_step]
        current_cci = self.data['CCI'].iloc[self.current_step]
        current_bb_upper = self.data['BB_upper'].iloc[self.current_step]
        current_bb_lower = self.data['BB_lower'].iloc[self.current_step]
        
        # Calculate position metrics
        position_value = self.shares * current_price
        position_ratio = position_value / self.total_value if self.total_value > 0 else 0
        unrealized_pnl = position_value - (self.shares * self.data['Close'].iloc[self.current_step - 1]) if self.current_step > 0 else 0
        
        return np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares,
            current_price / self.data['Close'].iloc[0],  # Normalized price
            current_returns,
            current_ma5 / current_price,
            current_ma20 / current_price,
            current_rsi / 100,  # Normalized RSI
            current_macd,
            current_volume / self.data['Volume'].mean(),  # Normalized volume
            current_volatility,
            current_atr / current_price,  # Normalized ATR
            current_obv / self.data['OBV'].max(),  # Normalized OBV
            current_vwap / current_price,
            current_adx / 100,  # Normalized ADX
            current_cci / 100,  # Normalized CCI
            current_bb_upper / current_price,
            current_bb_lower / current_price,
            position_ratio,
            unrealized_pnl / self.initial_balance,  # Normalized PnL
            len(self.trades) / 100  # Normalized trade count
        ], dtype=np.float32)
    
    def step(self, action):
        """Execute one time step within the environment with realistic trading conditions"""
        current_price = self.data['Close'].iloc[self.current_step]
        
        # Apply slippage
        if action == 1:  # Buy
            execution_price = current_price * (1 + self.slippage)
        elif action == 2:  # Sell
            execution_price = current_price * (1 - self.slippage)
        else:
            execution_price = current_price
        
        # Execute action with transaction fees
        reward = 0
        if action == 1:  # Buy
            if self.balance >= execution_price:
                shares_to_buy = int(self.balance / (execution_price * (1 + self.transaction_fee)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * execution_price * (1 + self.transaction_fee)
                    self.shares += shares_to_buy
                    self.balance -= cost
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'price': execution_price,
                        'shares': shares_to_buy,
                        'cost': cost
                    })
        elif action == 2:  # Sell
            if self.shares > 0:
                revenue = self.shares * execution_price * (1 - self.transaction_fee)
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': execution_price,
                    'shares': self.shares,
                    'revenue': revenue
                })
                self.balance += revenue
                self.shares = 0
        
        # Calculate reward (change in portfolio value)
        new_total_value = self.balance + (self.shares * current_price)
        reward = new_total_value - self.total_value
        self.total_value = new_total_value
        
        # Store portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'total_value': self.total_value,
            'balance': self.balance,
            'shares': self.shares,
            'price': current_price
        })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate additional info
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'shares': self.shares,
            'current_price': current_price,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history
        }
        
        return self._get_observation(), reward, done, False, info
    
    def render(self):
        """Render the environment with detailed information"""
        current_price = self.data['Close'].iloc[self.current_step]
        print(f'\nStep: {self.current_step}')
        print(f'Current Price: ${current_price:.2f}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Shares: {self.shares}')
        print(f'Total Value: ${self.total_value:.2f}')
        print(f'Return: {((self.total_value - self.initial_balance) / self.initial_balance * 100):.2f}%')
        print(f'Number of Trades: {len(self.trades)}')
        print('-------------------')
    
    def get_portfolio_history(self):
        """Get the complete portfolio history"""
        return pd.DataFrame(self.portfolio_history)
    
    def get_trade_history(self):
        """Get the complete trade history"""
        return pd.DataFrame(self.trades) 