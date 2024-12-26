import pytest
from click.testing import CliRunner
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from cli import cli, prepare_data, optimize, validate, report

@pytest.fixture
def runner():
    """Fixture per il CLI runner"""
    return CliRunner()

@pytest.fixture
def sample_data(tmp_path):
    """Fixture che crea dati di esempio"""
    # Genera prima i prezzi base
    base_price = 100 * (1 + np.random.randn(100) * 0.02)  # 2% di volatilità
    
    # Crea il timestamp come indice
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='h')
    
    # Genera i prezzi in modo coerente
    data = pd.DataFrame(index=timestamps)
    data['open'] = base_price * (1 + np.random.randn(100) * 0.001)
    data['close'] = base_price * (1 + np.random.randn(100) * 0.001)
    data['volume'] = np.random.uniform(1000000, 2000000, 100)
    
    # Calcola high e low in modo coerente
    daily_range = base_price * 0.02  # 2% di range giornaliero
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.randn(100) * daily_range)
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.randn(100) * daily_range)
    
    # Reset dell'indice per salvare il timestamp come colonna
    data = data.reset_index()
    data = data.rename(columns={'index': 'timestamp'})
    
    csv_path = tmp_path / "sample_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def strategy_results(tmp_path):
    """Fixture che crea risultati di esempio per una strategia"""
    results = {
        'strategy_type': 'options',
        'best_strategy': {
            'name': 'OptionsStrategy',
            'genes': {
                'rsi': {
                    'type': 'RSIGene',
                    'name': 'rsi',
                    'value': 14
                },
                'bollinger': {
                    'type': 'BollingerGene',
                    'name': 'bollinger',
                    'value': 20
                },
                'atr': {
                    'type': 'ATRGene',
                    'name': 'atr',
                    'value': 14
                }
            },
            'params': {
                'position_sizing': {
                    'method': 'fixed',
                    'base_size': 1.0,
                    'volatility_multiplier': 1.0
                },
                'entry_conditions': {
                    'rsi_thresholds': {
                        'oversold': 30,
                        'overbought': 70
                    },
                    'volatility_conditions': {
                        'min_atr': 0.01
                    },
                    'price_conditions': {
                        'bb_threshold': 0.5
                    }
                },
                'exit_conditions': {
                    'profit_target': 0.05,
                    'stop_loss': 0.02,
                    'time_decay_threshold': 0.5
                },
                'risk_management': {
                    'max_positions': 3,
                    'max_vega_exposure': 0.5,
                    'max_position_size': 2.0
                }
            }
        },
        'optimization_log': [
            {
                'generation': 0,
                'avg': 0.5,
                'std': 0.1,
                'min': 0.3,
                'max': 0.8
            },
            {
                'generation': 1,
                'avg': 0.6,
                'std': 0.1,
                'min': 0.4,
                'max': 0.9
            }
        ]
    }
    
    results_path = tmp_path / "strategy_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    return results_path

@pytest.fixture
def validation_results(tmp_path):
    """Fixture che crea risultati di validazione di esempio"""
    results = {
        'total_return': 0.15,
        'annual_return': 0.25,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.1,
        'win_rate': 0.6
    }
    
    results_path = tmp_path / "validation_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    return results_path

class TestCLI:
    def test_prepare_data(self, runner, sample_data, tmp_path):
        """Test del comando prepare-data"""
        output_dir = tmp_path / "processed"
        
        result = runner.invoke(prepare_data, [
            '--data-path', str(sample_data),
            '--output-dir', str(output_dir),
            '--train-ratio', '0.8'
        ])
        
        assert result.exit_code == 0
        assert (output_dir / 'train_data.parquet').exists()
        assert (output_dir / 'test_data.parquet').exists()

    def test_prepare_data_invalid_path(self, runner, tmp_path):
        """Test prepare-data con percorso non valido"""
        result = runner.invoke(prepare_data, [
            '--data-path', 'nonexistent.csv',
            '--output-dir', str(tmp_path)
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_optimize(self, runner, sample_data, tmp_path):
        """Test del comando optimize"""
        output_dir = tmp_path / "results"
        
        result = runner.invoke(optimize, args=['--strategy', 'options', '--data-path', str(sample_data), '--output-dir', str(output_dir)])
        if result.exit_code != 0:
            print(f"Error output: {result.output}")
        
        assert result.exit_code == 0
        assert any(file.suffix == '.yaml' for file in output_dir.iterdir())

    def test_optimize_invalid_strategy(self, runner, sample_data, tmp_path):
        """Test optimize con strategia non valida"""
        result = runner.invoke(optimize, [
            '--strategy', 'invalid',
            '--data-path', str(sample_data),
            '--output-dir', str(tmp_path)
        ])
        
        assert result.exit_code != 0

    def test_validate(self, runner, sample_data, strategy_results, tmp_path):
        """Test del comando validate"""
        output_dir = tmp_path / "validation"
        
        result = runner.invoke(validate, args=['--strategy-path', str(strategy_results), '--data-path', str(sample_data), '--output-dir', str(output_dir)])
        if result.exit_code != 0:
            print(f"Error output: {result.output}")
        
        assert result.exit_code == 0
        assert any(file.suffix == '.yaml' for file in output_dir.iterdir())

    def test_validate_invalid_strategy_path(self, runner, sample_data, tmp_path):
        """Test validate con percorso strategia non valido"""
        result = runner.invoke(validate, [
            '--strategy-path', 'nonexistent.yaml',
            '--data-path', str(sample_data),
            '--output-dir', str(tmp_path)
        ])
        
        assert result.exit_code != 0

    def test_report(self, runner, validation_results):
        """Test del comando report"""
        result = runner.invoke(report, [
            '--results-path', str(validation_results)
        ])
        
        assert result.exit_code == 0
        assert "Performance Report" in result.output
        assert "Total Return" in result.output

    def test_report_invalid_path(self, runner):
        """Test report con percorso non valido"""
        result = runner.invoke(report, [
            '--results-path', 'nonexistent.yaml'
        ])
        
        assert result.exit_code != 0

    def test_cli_help(self, runner):
        """Test del comando help"""
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Trading System CLI" in result.output
        
    def test_prepare_data_help(self, runner):
        """Test dell'help per prepare-data"""
        result = runner.invoke(prepare_data, ['--help'])
        
        assert result.exit_code == 0
        assert "Prepare and validate data" in result.output

    def test_all_commands_sequence(self, runner, sample_data, tmp_path):
        """Test della sequenza completa di comandi"""
        # 1. Prepare data
        processed_dir = tmp_path / "processed"
        result1 = runner.invoke(prepare_data, [
            '--data-path', str(sample_data),
            '--output-dir', str(processed_dir)
        ])
        assert result1.exit_code == 0
        
        # 2. Optimize strategy
        results_dir = tmp_path / "results"
        result2 = runner.invoke(optimize, args=['--strategy', 'options', '--data-path', str(processed_dir / 'train_data.parquet'), '--output-dir', str(results_dir)])
        if result2.exit_code != 0:
            print(f"Error output: {result2.output}")
        assert result2.exit_code == 0
        
        # Get the generated strategy file
        strategy_file = next(results_dir.glob('*.yaml'))
        
        # 3. Validate strategy
        validation_dir = tmp_path / "validation"
        result3 = runner.invoke(validate, ['--strategy-path', str(strategy_file), '--data-path', str(processed_dir / 'test_data.parquet'), '--output-dir', str(validation_dir)])
        assert result3.exit_code == 0
        
        # Get the validation results file
        validation_file = next(validation_dir.glob('*.yaml'))
        
        # 4. Generate report
        result4 = runner.invoke(report, [
            '--results-path', str(validation_file)
        ])
        assert result4.exit_code == 0
