"""
Advanced evaluation metrics and walk-forward analysis
Phase 1: Foundation implementation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedEvaluator:
    """Comprehensive evaluation metrics for trading models"""
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 prices: np.ndarray, initial_capital: float = 10000) -> Dict:
        """Calculate comprehensive trading performance metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except:
            roc_auc = 0.5
        
        # Trading strategy metrics
        strategy_returns = self._calculate_strategy_returns(y_true, y_pred, prices)
        buy_hold_returns = (prices[-1] - prices[0]) / prices[0]
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(strategy_returns)
        max_drawdown = self._calculate_max_drawdown(strategy_returns)
        calmar_ratio = self._calculate_calmar_ratio(strategy_returns)
        
        # Win rate and profit factor
        win_rate = self._calculate_win_rate(strategy_returns)
        profit_factor = self._calculate_profit_factor(strategy_returns)
        
        return {
            'classification': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'trading': {
                'total_return': strategy_returns[-1] - 1,
                'buy_hold_return': buy_hold_returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
        }
    
    def _calculate_strategy_returns(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  prices: np.ndarray) -> np.ndarray:
        """Calculate cumulative returns from trading strategy"""
        
        # Simple long-only strategy based on predictions
        positions = np.where(y_pred == 1, 1, 0)  # Long when predicted positive
        
        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = np.array([0])  # Initial return
        
        for i in range(1, len(positions)):
            if positions[i-1] == 1:
                strategy_returns = np.append(strategy_returns, price_returns[i-1])
            else:
                strategy_returns = np.append(strategy_returns, 0)
        
        # Cumulative returns
        cumulative_returns = np.cumprod(1 + strategy_returns)
        
        return cumulative_returns
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns[-1] - 1 - risk_free_rate * (len(returns) / 252)
        volatility = np.std(np.diff(returns))
        
        if volatility == 0:
            return 0
        
        return excess_returns / volatility
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(returns)
        drawdown = (returns - peak) / peak
        return np.min(drawdown)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = (returns[-1] ** (252 / len(returns))) - 1
        max_dd = abs(self._calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return 0
        
        return annual_return / max_dd
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate of trades"""
        daily_returns = np.diff(returns)
        winning_days = np.sum(daily_returns > 0)
        total_days = len(daily_returns)
        
        if total_days == 0:
            return 0
        
        return winning_days / total_days
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        daily_returns = np.diff(returns)
        gross_profit = np.sum(daily_returns[daily_returns > 0])
        gross_loss = abs(np.sum(daily_returns[daily_returns < 0]))
        
        if gross_loss == 0:
            return float('inf')
        
        return gross_profit / gross_loss
    
    def plot_evaluation_results(self, metrics: Dict, save_path: str = None):
        """Create comprehensive evaluation plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # Classification metrics
        ax1 = axes[0, 0]
        class_metrics = metrics['classification']
        bars = ax1.bar(class_metrics.keys(), class_metrics.values())
        ax1.set_title('Classification Metrics')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Trading metrics
        ax2 = axes[0, 1]
        trading_metrics = metrics['trading']
        trading_df = pd.DataFrame([trading_metrics]).T
        trading_df.plot(kind='bar', ax=ax2, legend=False)
        ax2.set_title('Trading Performance Metrics')
        ax2.tick_params(axis='x', rotation=45)
        
        # Confusion matrix
        ax3 = axes[1, 0]
        # This would need actual y_true, y_pred
        ax3.text(0.5, 0.5, 'Confusion Matrix\n(Requires predictions)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Confusion Matrix')
        
        # Returns comparison
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Returns Comparison\n(Requires predictions)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Strategy vs Buy & Hold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class WalkForwardAnalysis:
    """Walk-forward analysis for realistic backtesting"""
    
    def __init__(self, model_trainer, data: pd.DataFrame, 
                 train_window: int = 252, test_window: int = 63):
        self.model_trainer = model_trainer
        self.data = data
        self.train_window = train_window  # 1 year
        self.test_window = test_window    # 3 months
        self.results = []
        
    def run_walk_forward(self) -> Dict:
        """Execute complete walk-forward analysis"""
        
        total_days = len(self.data)
        current_idx = self.train_window
        
        while current_idx + self.test_window <= total_days:
            # Training period
            train_start = current_idx - self.train_window
            train_end = current_idx
            
            # Testing period
            test_start = current_idx
            test_end = current_idx + self.test_window
            
            # Prepare data
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            # Train model
            data_dict = self.model_trainer.prepare_data(train_data)
            training_results = self.model_trainer.train_model(data_dict, epochs=50)
            
            # Make predictions on test set
            test_features = self.model_trainer.feature_engineer.create_sequences(
                self.model_trainer.feature_engineer.add_technical_indicators(test_data)
            )
            
            if len(test_features[0]) > 0:
                predictions = self.model_trainer.predict(test_features[0])
                
                # Calculate metrics
                evaluator = AdvancedEvaluator()
                metrics = evaluator.calculate_trading_metrics(
                    test_features[1], 
                    (predictions > 0.5).astype(int),
                    test_data['Close'].values
                )
                
                self.results.append({
                    'period': f"{test_start}-{test_end}",
                    'metrics': metrics,
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                })
            
            current_idx += self.test_window
        
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        """Summarize walk-forward analysis results"""
        
        if not self.results:
            return {}
        
        # Extract key metrics
        returns = [r['metrics']['trading']['total_return'] for r in self.results]
        sharpe_ratios = [r['metrics']['trading']['sharpe_ratio'] for r in self.results]
        max_drawdowns = [r['metrics']['trading']['max_drawdown'] for r in self.results]
        
        summary = {
            'total_periods': len(self.results),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'win_rate': np.mean([r > 0 for r in returns]),
            'period_results': self.results
        }
        
        return summary
    
    def plot_walk_forward_results(self, save_path: str = None):
        """Visualize walk-forward analysis results"""
        
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Analysis Results', fontsize=16)
        
        # Returns over time
        ax1 = axes[0, 0]
        returns = [r['metrics']['trading']['total_return'] for r in self.results]
        ax1.plot(returns, marker='o')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax1.set_title('Returns by Period')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Return')
        
        # Sharpe ratios
        ax2 = axes[0, 1]
        sharpe_ratios = [r['metrics']['trading']['sharpe_ratio'] for r in self.results]
        ax2.plot(sharpe_ratios, marker='s', color='green')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('Sharpe Ratios by Period')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Sharpe Ratio')
        
        # Max drawdowns
        ax3 = axes[1, 0]
        max_drawdowns = [r['metrics']['trading']['max_drawdown'] for r in self.results]
        ax3.plot(max_drawdowns, marker='^', color='orange')
        ax3.set_title('Maximum Drawdowns by Period')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Max Drawdown')
        
        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        Walk-Forward Analysis Summary:
        
        Total Periods: {len(self.results)}
        Average Return: {np.mean(returns):.2%}
        Average Sharpe: {np.mean(sharpe_ratios):.2f}
        Average Max DD: {np.mean(max_drawdowns):.2%}
        Win Rate: {np.mean([r > 0 for r in returns]):.1%}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class ModelRegistry:
    """Model versioning and A/B testing registry"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = registry_path
        import os
        os.makedirs(registry_path, exist_ok=True)
        
    def register_model(self, model, metrics: Dict, version: str = None):
        """Register a trained model with metrics"""
        if version is None:
            version = f"v{len(os.listdir(self.registry_path)) + 1}"
        
        model_info = {
            'version': version,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': type(model).__name__
        }
        
        # Save model info
        info_path = os.path.join(self.registry_path, f"{version}_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save model
        model_path = os.path.join(self.registry_path, f"{version}_model.pth")
        torch.save(model.state_dict(), model_path)
        
        return version
    
    def get_best_model(self, metric: str = 'sharpe_ratio') -> str:
        """Get the best performing model version"""
        versions = []
        
        for file in os.listdir(self.registry_path):
            if file.endswith('_info.json'):
                with open(os.path.join(self.registry_path, file), 'r') as f:
                    info = json.load(f)
                    versions.append(info)
        
        if not versions:
            return None
        
        # Sort by specified metric
        best_version = max(versions, key=lambda x: x['metrics']['trading'][metric])
        return best_version['version']

if __name__ == "__main__":
    # Example usage
    from src.lstm_model import MarketModelTrainer
    
    # Load sample data
    import yfinance as yf
    data = yf.download("AAPL", period="2y")
    
    # Initialize trainer
    trainer = MarketModelTrainer(model_type='lstm')
    
    # Run walk-forward analysis
    wfa = WalkForwardAnalysis(trainer, data)
    results = wfa.run_walk_forward()
    
    print("Walk-forward analysis completed!")
    print(f"Average Sharpe Ratio: {results['avg_sharpe']:.2f}")
    print(f"Win Rate: {results['win_rate']:.1%}")
