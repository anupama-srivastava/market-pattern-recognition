#!/usr/bin/env python3
"""
Production-ready training script for Phase 1
Advanced Market Pattern Recognition System
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.lstm_model import MarketModelTrainer, LSTMMarketPredictor, GRUMarketPredictor
from src.advanced_evaluation import AdvancedEvaluator, WalkForwardAnalysis, ModelRegistry
from src.data_collection import StockDataCollector
from src.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phase1Trainer:
    """Production-ready training pipeline for Phase 1"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.model_registry = ModelRegistry()
        self.evaluator = AdvancedEvaluator()
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load training configuration"""
        default_config = {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
            "period": "2y",
            "model_type": "lstm",
            "sequence_length": 60,
            "prediction_horizon": 5,
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "test_size": 0.2,
            "walk_forward": {
                "enabled": True,
                "train_window": 252,
                "test_window": 63
            },
            "output_dir": "models/phase1"
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download training data for all symbols"""
        logger.info("Downloading market data...")
        
        collector = StockDataCollector()
        data = {}
        
        for symbol in self.config['symbols']:
            logger.info(f"Downloading {symbol}...")
            df = collector.download_stock_data(symbol, self.config['period'])
            if df is not None:
                data[symbol] = df
        
        logger.info(f"Successfully downloaded data for {len(data)} symbols")
        return data
    
    def train_single_model(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Train model for a single symbol"""
        logger.info(f"Training model for {symbol}...")
        
        # Initialize trainer
        trainer = MarketModelTrainer(
            model_type=self.config['model_type']
        )
        
        # Prepare data
        data_dict = trainer.prepare_data(
            data,
            sequence_length=self.config['sequence_length'],
            batch_size=self.config['batch_size'],
            test_size=self.config['test_size']
        )
        
        # Train model
        training_results = trainer.train_model(
            data_dict,
            epochs=self.config['epochs'],
            learning_rate=self.config['learning_rate']
        )
        
        # Evaluate model
        model = training_results['model']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # Make predictions
        predictions = model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self.evaluator.calculate_trading_metrics(
            y_test, y_pred, data['Close'].iloc[-len(y_test):].values
        )
        
        # Save model
        model_path = os.path.join(
            self.config['output_dir'], 
            f"{symbol}_{self.config['model_type']}_model.pth"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        # Register model
        version = self.model_registry.register_model(
            model, metrics, version=f"{symbol}_{self.config['model_type']}_v1"
        )
        
        return {
            'symbol': symbol,
            'model_path': model_path,
            'metrics': metrics,
            'version': version,
            'training_results': training_results
        }
    
    def run_walk_forward_analysis(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Run walk-forward analysis for a symbol"""
        logger.info(f"Running walk-forward analysis for {symbol}...")
        
        trainer = MarketModelTrainer(model_type=self.config['model_type'])
        wfa = WalkForwardAnalysis(
            trainer, 
            data,
            train_window=self.config['walk_forward']['train_window'],
            test_window=self.config['walk_forward']['test_window']
        )
        
        results = wfa.run_walk_forward()
        
        # Save walk-forward results
        wfa_path = os.path.join(
            self.config['output_dir'], 
            f"{symbol}_walk_forward_results.json"
        )
        with open(wfa_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def train_all_models(self) -> Dict:
        """Train models for all symbols"""
        logger.info("Starting Phase 1 training...")
        
        # Download data
        data = self.download_data()
        
        results = {
            'training_results': {},
            'walk_forward_results': {},
            'summary': {}
        }
        
        # Train models
        for symbol, df in data.items():
            try:
                # Train single model
                train_result = self.train_single_model(symbol, df)
                results['training_results'][symbol] = train_result
                
                # Run walk-forward analysis if enabled
                if self.config['walk_forward']['enabled']:
                    wfa_result = self.run_walk_forward_analysis(symbol, df)
                    results['walk_forward_results'][symbol] = wfa_result
                
                logger.info(f"✅ Completed training for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error training {symbol}: {str(e)}")
                continue
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate training summary"""
        summary = {
            'total_symbols': len(results['training_results']),
            'successful_models': len(results['training_results']),
            'average_accuracy': np.mean([
                r['metrics']['classification']['accuracy'] 
                for r in results['training_results'].values()
            ]),
            'average_sharpe': np.mean([
                r['metrics']['trading']['sharpe_ratio'] 
                for r in results['training_results'].values()
            ]),
            'best_performer': max(
                results['training_results'].items(),
                key=lambda x: x[1]['metrics']['trading']['sharpe_ratio']
            )[0] if results['training_results'] else None
        }
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            self.config['output_dir'], 
            f"training_results_{timestamp}.json"
        )
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def create_training_report(self, results: Dict):
        """Create comprehensive training report"""
        report_path = os.path.join(self.config['output_dir'], "training_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Phase 1 Training Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Total Symbols: {results['summary']['total_symbols']}\n")
            f.write(f"- Successful Models: {results['summary']['successful_models']}\n")
            f.write(f"- Average Accuracy: {results['summary']['average_accuracy']:.3f}\n")
            f.write(f"- Average Sharpe Ratio: {results['summary']['average_sharpe']:.3f}\n")
            f.write(f"- Best Performer: {results['summary']['best_performer']}\n\n")
            
            f.write("## Model Performance by Symbol\n")
            for symbol, result in results['training_results'].items():
                f.write(f"\n### {symbol}\n")
                f.write(f"- Accuracy: {result['metrics']['classification']['accuracy']:.3f}\n")
                f.write(f"- Sharpe Ratio: {result['metrics']['trading']['sharpe_ratio']:.3f}\n")
                f.write(f"- Max Drawdown: {result['metrics']['trading']['max_drawdown']:.3f}\n")
                f.write(f"- Model Path: {result['model_path']}\n")
        
        logger.info(f"Training report saved to {report_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Phase 1 Training Script')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to train')
    parser.add_argument('--model-type', type=str, choices=['lstm', 'gru'], 
                       default='lstm', help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--output-dir', type=str, default='models/phase1', 
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config = {}
    if args.symbols:
        config['symbols'] = args.symbols
    if args.model_type:
        config['model_type'] = args.model_type
    if args.epochs:
        config['epochs'] = args.epochs
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Create temporary config file
    config_path = None
    if config:
        config_path = 'temp_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    # Initialize trainer
    trainer = Phase1Trainer(config_path or args.config)
    
    # Run training
    results = trainer.train_all_models()
    
    # Create report
    trainer.create_training_report(results)
    
    # Clean up temporary config
    if config_path and os.path.exists(config_path):
        os.remove(config_path)
    
    print("\n🎉 Phase 1 training completed successfully!")
    print(f"📊 Summary: {results['summary']}")

if __name__ == "__main__":
    main()
