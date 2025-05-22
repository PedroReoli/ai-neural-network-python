import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CustomNetwork(nn.Module):
    def __init__(self, features_dim: int):
        super(CustomNetwork, self).__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 actions: hold, buy, sell
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x)
        return self.policy_net(shared_features), self.value_net(shared_features)

class TradingAgent:
    def __init__(self, env, model_type='ppo'):
        self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        self.model_type = model_type
        self.model = None
        self.training_history = []
        
    def train(self, total_timesteps=100000, eval_freq=10000):
        """Train the agent using the specified model type"""
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: Monitor(self.env.envs[0])])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./best_model',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Create custom callback for tracking
        tracking_callback = TradingCallback()
        
        # Initialize model based on type
        if self.model_type == 'ppo':
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[256, 128, 64],
                        vf=[256, 128, 64]
                    )
                )
            )
        elif self.model_type == 'a2c':
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=0.0007,
                n_steps=5,
                gamma=0.99,
                verbose=1
            )
        elif self.model_type == 'dqn':
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=0.0001,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=64,
                gamma=0.99,
                verbose=1
            )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, tracking_callback]
        )
        
        self.training_history = tracking_callback.portfolio_values
        
    def predict(self, observation, deterministic=True):
        """Make a prediction based on the current observation"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path):
        """Save the trained model and environment normalizer"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(f"{path}_model")
        self.env.save(f"{path}_env.pkl")
    
    def load(self, path):
        """Load a trained model and environment normalizer"""
        self.model = PPO.load(f"{path}_model", env=self.env)
        self.env = VecNormalize.load(f"{path}_env.pkl", env=self.env)
    
    def evaluate_performance(self, test_env) -> Dict:
        """Evaluate the agent's performance"""
        obs = test_env.reset()
        done = False
        total_reward = 0
        trades = []
        
        while not done:
            action = self.predict(obs)
            obs, reward, done, _, info = test_env.step(action)
            total_reward += reward
            
            if 'trades' in info and info['trades']:
                trades.extend(info['trades'])
        
        # Calculate performance metrics
        portfolio_history = test_env.get_portfolio_history()
        trade_history = pd.DataFrame(trades)
        
        returns = portfolio_history['total_value'].pct_change().dropna()
        
        metrics = {
            'total_return': (portfolio_history['total_value'].iloc[-1] / portfolio_history['total_value'].iloc[0] - 1) * 100,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (portfolio_history['total_value'] / portfolio_history['total_value'].cummax() - 1).min() * 100,
            'num_trades': len(trades),
            'win_rate': len(trade_history[trade_history['revenue'] > trade_history['cost']]) / len(trade_history) if len(trade_history) > 0 else 0,
            'avg_trade_return': trade_history['revenue'].sum() / trade_history['cost'].sum() - 1 if len(trade_history) > 0 else 0
        }
        
        return metrics
    
    def plot_performance(self, test_env):
        """Plot the agent's performance"""
        obs = test_env.reset()
        done = False
        portfolio_values = []
        
        while not done:
            action = self.predict(obs)
            obs, _, done, _, _ = test_env.step(action)
            portfolio_values.append(test_env.total_value)
        
        # Create performance plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value plot
        ax1.plot(portfolio_values)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Trading Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Returns distribution plot
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sns.histplot(returns, ax=ax2, kde=True)
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Returns')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig

class TradingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.portfolio_values = []
        self.rewards = []
        self.trades = []
    
    def _on_step(self):
        """Called at each step during training"""
        # Get the current portfolio value
        env = self.training_env.envs[0]
        portfolio_value = env.total_value
        
        # Store the values
        self.portfolio_values.append(portfolio_value)
        self.rewards.append(self.locals['rewards'][0])
        
        # Store trades if any
        if 'trades' in self.locals['infos'][0]:
            self.trades.extend(self.locals['infos'][0]['trades'])
        
        return True 