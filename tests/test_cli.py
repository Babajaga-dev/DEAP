import pytest
from click.testing import CliRunner
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
from src.cli import cli, optimize, backtest, config

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def sample_data(tmp_path):
    """Crea file di dati di test"""
    # Crea dati nel formato corretto
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'open': np.random.uniform(100, 110, 10).astype(np.float64),
        'high': np.random.uniform(110, 120, 10).astype(np.float64),
        'low': np.random.uniform(90, 100, 10).astype(np.float64),
        'close': np.random.uniform(100, 110, 10).astype(np.float64),
        'volume': np.random.uniform(1000, 2000, 10).astype(np.float64)
    })
    
    # Assicura che high sia sempre maggiore di low
    data['high'] = data[['high', 'low']].max(axis=1) + 1
    data['low'] = data[['high', 'low']].min(axis=1)
    
    # Salva come parquet per mantenere i tipi di dati
    data_file = tmp_path / "test_data.parquet"
    data.to_parquet(data_file)
    return str(data_file)

@pytest.fixture
def sample_strategy(tmp_path):
    """Crea file strategia di test"""
    strategy_file = tmp_path / "test_strategy.json"
    strategy_data = {
        "name": "TrendMomentumStrategy",
        "genes": {},
        "params": {
            "position_sizing": {"method": "fixed", "base_size": 1.0},
                    "entry_conditions": {
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "rsi_window": 14
                    },
                    "exit_conditions": {
                        "take_profit": 0.05,
                        "stop_loss": 0.02
                    },
                    "risk_management": {
                        "position_size": 1.0,
                        "max_positions": 1
                    }
        }
    }
    strategy_file.write_text(json.dumps(strategy_data))
    return str(strategy_file)

@pytest.fixture
def sample_config(tmp_path):
    """Crea file configurazione di test"""
    # Crea directory config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Crea configurazioni necessarie
    configs = {
        "genetic.yaml": {
            "optimization": {
                "population_size": 50,
                "generations": 100,
                "crossover_prob": 0.7,
                "mutation_prob": 0.2
            }
        },
        "data.yaml": {
            "data": {
                "market": {
                    "required_columns": ["open", "high", "low", "close", "volume"],
                    "price_decimals": 2,
                    "volume_decimals": 0
                },
                "preprocessing": {
                    "handle_missing": {"method": "forward_fill", "max_consecutive_missing": 5},
                    "handle_outliers": {"method": "zscore", "threshold": 3},
                    "volume_filter": {"min_volume": 0}
                },
                "validation": {
                    "price_checks": {
                        "high_low": True,
                        "open_close_range": True,
                        "zero_prices": False
                    },
                    "volume_checks": {"max_volume": None},
                    "timestamp_checks": {"duplicates": "error"}
                }
            }
        },
        "strategies.yaml": {
            "strategies": {
                "trendmomentum": {
                    "name": "TrendMomentumStrategy",
                    "position_sizing": {"method": "fixed", "base_size": 1.0},
                    "indicators": ["sma", "rsi"],
            "entry_conditions": {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "rsi_window": 14
            },
            "exit_conditions": {
                "take_profit": 0.05,
                "stop_loss": 0.02
            },
            "risk_management": {
                "position_size": 1.0,
                "max_positions": 1
            }
                },
                "general": {
                    "risk_free_rate": 0.02,
                    "transaction_costs": 0.001
                }
            }
        },
        "indicators.yaml": {
            "indicators": {
                "sma": {
                    "window": {"min": 5, "max": 200, "default": 20},
                    "price": {"options": ["close", "open", "high", "low"]}
                },
                "rsi": {
                    "window": {"min": 2, "max": 100, "default": 14},
                    "overbought": {"min": 50, "max": 100, "default": 70},
                    "oversold": {"min": 0, "max": 50, "default": 30}
                }
            }
        },
        "backtest.yaml": {
            "backtest": {
                "metrics": {
                    "basic": ["total_return", "annual_return", "sharpe_ratio", "max_drawdown"],
                    "advanced": ["sortino_ratio", "calmar_ratio"]
                },
                "costs": {
                    "commission": {"value": 0.001, "type": "percentage"},
                    "slippage": {"value": 0.0001, "type": "percentage"}
                }
            }
        }
    }
    
    # Salva le configurazioni
    for filename, config in configs.items():
        config_file = config_dir / filename
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
    
    return config_dir

def test_cli_help(runner):
    """Test help command"""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'CLI per ottimizzazione genetica' in result.output

def test_optimize_command(runner, sample_data, sample_config, monkeypatch):
    """Test optimize command"""
    # Imposta la directory di config temporanea
    monkeypatch.setenv('CONFIG_DIR', str(sample_config))
    
    result = runner.invoke(cli, [
        'optimize',
        '--strategy', 'trend',
        '--data', sample_data,
        '--generations', '10',
        '--population', '20'
    ])
    # Verifica solo il codice di uscita dato che i log vanno su stderr
    assert result.exit_code == 0

def test_backtest_command(runner, sample_data, sample_strategy, sample_config, monkeypatch):
    """Test backtest command"""
    # Imposta la directory di config temporanea
    monkeypatch.setenv('CONFIG_DIR', str(sample_config))
    
    result = runner.invoke(cli, [
        'backtest',
        '--strategy', 'trend',
        '--data', sample_data,
        '--model', sample_strategy
    ])
    # Verifica solo il codice di uscita dato che i log vanno su stderr
    assert result.exit_code == 0

def test_config_show(runner, sample_config, monkeypatch):
    """Test config show command"""
    # Imposta la directory di config temporanea
    monkeypatch.setenv('CONFIG_DIR', str(sample_config))
    
    result = runner.invoke(cli, [
        'config', 'show',
        '--type', 'genetic'
    ])
    assert result.exit_code == 0

def test_config_edit(runner, sample_config, monkeypatch):
    """Test config edit command"""
    # Imposta la directory di config temporanea
    monkeypatch.setenv('CONFIG_DIR', str(sample_config))
    
    config_file = sample_config / "genetic.yaml"
    result = runner.invoke(cli, [
        'config', 'edit',
        '--type', 'genetic',
        '--param', 'optimization.population_size',
        '--value', '100'
    ])
    assert result.exit_code == 0
    
    # Verifica modifica
    with open(config_file) as f:
        config = yaml.safe_load(f)
        assert config['optimization']['population_size'] == 100

def test_examples_command(runner):
    """Test examples command"""
    result = runner.invoke(cli, ['examples'])
    assert result.exit_code == 0
    assert 'Esempi di utilizzo' in result.output

def test_invalid_strategy(runner, sample_data):
    """Test errore con strategia invalida"""
    result = runner.invoke(cli, [
        'optimize',
        '--strategy', 'invalid',
        '--data', sample_data
    ])
    assert result.exit_code != 0

def test_missing_data_file(runner):
    """Test errore con file dati mancante"""
    result = runner.invoke(cli, [
        'optimize',
        '--strategy', 'trend',
        '--data', 'nonexistent.csv'
    ])
    assert result.exit_code != 0

def test_invalid_config_type(runner):
    """Test errore con tipo configurazione invalido"""
    result = runner.invoke(cli, [
        'config', 'show',
        '--type', 'invalid'
    ])
    assert result.exit_code != 0
