import pytest
import os
import yaml
from pathlib import Path
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a temporary config directory with test files"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create test configuration files
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
                }
            }
        },
        "strategies.yaml": {
            "strategies": {
                "trendmomentum": {
                    "name": "TrendMomentumStrategy",
                    "indicators": ["sma", "rsi"],
                    "position_sizing": {"method": "fixed", "base_size": 1.0}
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
                    "window": {"min": 5, "max": 200, "default": 20}
                },
                "rsi": {
                    "window": {"min": 2, "max": 100, "default": 14}
                }
            }
        },
        "backtest.yaml": {
            "backtest": {
                "metrics": ["total_return", "sharpe_ratio", "max_drawdown"],
                "costs": {
                    "commission": 0.001,
                    "slippage": 0.0001
                }
            }
        }
    }
    
    for filename, config in configs.items():
        config_file = config_dir / filename
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
            
    return config_dir

@pytest.fixture
def config_loader(mock_config_dir, monkeypatch):
    """Create a ConfigLoader instance with mock config directory"""
    monkeypatch.setenv('CONFIG_DIR', str(mock_config_dir))
    return ConfigLoader()

class TestConfigLoader:
    def test_init_with_env_var(self, mock_config_dir, monkeypatch):
        """Test initialization with environment variable"""
        monkeypatch.setenv('CONFIG_DIR', str(mock_config_dir))
        config_loader = ConfigLoader()
        assert config_loader.config_dir == str(mock_config_dir)

    def test_init_with_default(self, monkeypatch):
        """Test initialization with default path"""
        monkeypatch.delenv('CONFIG_DIR', raising=False)
        config_loader = ConfigLoader()
        assert 'config' in config_loader.config_dir

    def test_load_genetic_config(self, config_loader):
        """Test loading genetic configuration"""
        config = config_loader.get_genetic_config()
        assert isinstance(config, dict)
        assert 'population_size' in config
        assert config['population_size'] == 50

    def test_load_strategy_config(self, config_loader):
        """Test loading strategy configuration"""
        config = config_loader.get_strategy_config('trendmomentum')
        assert isinstance(config, dict)
        assert config['name'] == 'TrendMomentumStrategy'
        assert 'indicators' in config
        assert 'position_sizing' in config

    def test_load_indicator_config(self, config_loader):
        """Test loading indicator configuration"""
        config = config_loader.get_indicator_config('sma')
        assert isinstance(config, dict)
        assert 'window' in config
        assert config['window']['min'] == 5
        assert config['window']['max'] == 200

    def test_load_backtest_config(self, config_loader):
        """Test loading backtest configuration"""
        config = config_loader.get_backtest_config()
        assert isinstance(config, dict)
        assert 'metrics' in config
        assert 'costs' in config

    def test_get_risk_params(self, config_loader):
        """Test getting risk parameters"""
        params = config_loader.get_risk_params()
        assert isinstance(params, dict)
        assert 'risk_free_rate' in params
        assert 'transaction_costs' in params

    def test_validate_indicator_params(self, config_loader):
        """Test indicator parameter validation"""
        valid_params = {'window': {'default': 20}}
        assert config_loader.validate_indicator_params('sma', valid_params)

        with pytest.raises(ValueError):
            invalid_params = {'window': {'default': 1}}  # Below min
            config_loader.validate_indicator_params('sma', invalid_params)

    def test_get_all_strategies(self, config_loader):
        """Test getting list of all strategies"""
        strategies = config_loader.get_all_strategies()
        assert isinstance(strategies, list)
        assert 'trendmomentum' in strategies

    def test_get_all_indicators(self, config_loader):
        """Test getting list of all indicators"""
        indicators = config_loader.get_all_indicators()
        assert isinstance(indicators, list)
        assert 'sma' in indicators
        assert 'rsi' in indicators

    def test_update_config_value(self, config_loader):
        """Test updating configuration value"""
        config_loader.update_config_value('genetic', 'optimization.population_size', 100)
        config = config_loader.get_genetic_config()
        assert config['population_size'] == 100

    def test_clear_cache(self, config_loader):
        """Test clearing configuration cache"""
        # Load some configs to cache
        config_loader.get_genetic_config()
        config_loader.get_backtest_config()
        
        # Clear cache
        config_loader.clear_cache()
        assert len(config_loader._cached_configs) == 0

    def test_invalid_config_file(self, mock_config_dir, monkeypatch):
        """Test handling of invalid configuration file"""
        # Create an invalid YAML file
        invalid_file = mock_config_dir / "genetic.yaml"
        invalid_file.write_text("invalid: yaml: content:")
        
        monkeypatch.setenv('CONFIG_DIR', str(mock_config_dir))
        config_loader = ConfigLoader()
        
        with pytest.raises(ValueError):
            config_loader.load_config("genetic")

    def test_nonexistent_config_file(self, config_loader):
        """Test handling of nonexistent configuration file"""
        with pytest.raises(FileNotFoundError):
            config_loader.load_config("nonexistent")

    def test_nonexistent_indicator(self, config_loader):
        """Test handling of nonexistent indicator"""
        with pytest.raises(ValueError):
            config_loader.get_indicator_config("nonexistent")

    def test_nonexistent_strategy(self, config_loader):
        """Test handling of nonexistent strategy"""
        with pytest.raises(ValueError):
            config_loader.get_strategy_config("nonexistent")
