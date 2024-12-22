import click
import pandas as pd
import numpy as np
from numpy import float32, float64
from pathlib import Path
from typing import Dict, Any
import yaml
import logging
from datetime import datetime

from src.utils.config_loader import ConfigLoader
from src.data.data_loader import DataLoader
from src.strategies.trend_momentum_strategy import TrendMomentumStrategy
from src.strategies.options_strategy import OptionsStrategy
from src.optimization.genetic_optimizer import GeneticOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STRATEGY_CLASSES = {
    'trend_momentum': TrendMomentumStrategy,
    'options': OptionsStrategy
}

@click.group()
def cli():
    """Trading System CLI"""
    pass

@cli.command()
@click.option('--data-path', type=click.Path(exists=True), required=True, 
              help='Path to historical data CSV file')
@click.option('--output-dir', type=click.Path(), default='processed_data',
              help='Directory for processed data')
@click.option('--train-ratio', type=float, default=0.8,
              help='Ratio of data to use for training')
def prepare_data(data_path: str, output_dir: str, train_ratio: float):
    """Prepare and validate data for the trading system"""
    try:
        config_loader = ConfigLoader()
        data_loader = DataLoader(config_loader)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load and process data
        logger.info(f"Loading data from {data_path}")
        data = data_loader.load_csv(data_path)
        
        # Split data
        logger.info("Splitting data into train/test sets")
        train_data, test_data = data_loader.get_train_test_split(train_ratio=train_ratio)
        
        # Save processed data
        train_path = output_path / 'train_data.parquet'
        test_path = output_path / 'test_data.parquet'
        
        data_loader.save_data(train_data, str(train_path))
        data_loader.save_data(test_data, str(test_path))
        
        logger.info(f"Data prepared and saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--strategy', type=click.Choice(['trend_momentum', 'options']), 
              required=True, help='Strategy type to optimize')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to training data')
@click.option('--output-dir', type=click.Path(), default='results',
              help='Directory for optimization results')
def optimize(strategy: str, data_path: str, output_dir: str):
    """Optimize a trading strategy using genetic algorithm"""
    try:
        config_loader = ConfigLoader()
        
        # Initialize optimizer
        strategy_class = STRATEGY_CLASSES[strategy]
        optimizer = GeneticOptimizer(
            strategy_class=strategy_class,
            config_loader=config_loader
        )
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        optimizer.load_data(data_path)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Run optimization
        logger.info("Starting optimization...")
        best_strategy, logbook = optimizer.optimize(
            callback=lambda gen, stats: logger.info(f"Generation {gen}: {stats}")
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"{strategy}_results_{timestamp}.yaml"
        
        # Converti i valori numpy in float Python standard
        def convert_numpy_values(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_values(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_values(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj

        results = {
            'strategy_type': strategy,
            'best_strategy': best_strategy.to_dict(),
            'optimization_log': convert_numpy_values(logbook)
        }
        
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
            
        logger.info(f"Optimization results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--strategy-path', type=click.Path(exists=True), required=True,
              help='Path to optimized strategy results')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to test data')
@click.option('--output-dir', type=click.Path(), default='validation',
              help='Directory for validation results')
def validate(strategy_path: str, data_path: str, output_dir: str):
    """Validate an optimized strategy on test data"""
    try:
        config_loader = ConfigLoader()
        data_loader = DataLoader(config_loader)
        
        # Load strategy results
        with open(strategy_path, 'r') as f:
            strategy_results = yaml.safe_load(f)
        
        strategy_type = strategy_results['strategy_type']
        strategy_class = STRATEGY_CLASSES[strategy_type]
        
        # Initialize strategy from saved state
        strategy = strategy_class(config_loader)
        strategy.from_dict(strategy_results['best_strategy'])
        
        # Load test data
        test_data = data_loader.load_csv(data_path)
        
        # Run validation
        logger.info("Running validation...")
        validation_results = strategy.evaluate(test_data)
        
        # Save validation results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"validation_results_{timestamp}.yaml"
        
        with open(results_file, 'w') as f:
            yaml.dump(validation_results, f)
            
        logger.info(f"Validation results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--results-path', type=click.Path(exists=True), required=True,
              help='Path to validation results')
def report(results_path: str):
    """Generate performance report from validation results"""
    try:
        with open(results_path, 'r') as f:
            results = yaml.safe_load(f)
            
        click.echo("\nPerformance Report:")
        click.echo("==================")
        
        # Print metrics in a formatted way
        for metric, value in results.items():
            if isinstance(value, float):
                click.echo(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            else:
                click.echo(f"{metric.replace('_', ' ').title()}: {value}")
                
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()
