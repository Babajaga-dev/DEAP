import pytest
import os
import yaml
from pathlib import Path
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a temporary config directory with all required files"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create mock configs
    configs = {
        "indicators.yaml": {
            "indicators": {
                "rsi": {
                    "name": "Relative Strength Index",
                    "period": {
                        "min": 2,
                        "max": 50,
                        "default": 14
                    }
                }
            }
        },
        "genetic.yaml": {
            "optimization": {
                "population_size": 50,
                "generations": 100
            }
        },
        "strategies.yaml": {
            "strategies": {
                "general": {
                    "risk_free_rate": 0.02,
                    "transaction_costs": 0.001
                },
                "trend_momentum": {
                    "name": "Trend Momentum Strategy",
                    "indicators": ["rsi", "macd"]
                }
            }
        },
        "backtest.yaml": {
            "backtest": {
                "initial_capital": 100000,
                "currency": "USD"
            }
        }
    }
    
    for file_name, content in configs.items():
        file_path = config_dir / file_name
        with open(file_path, 'w') as f:
            yaml.dump(content, f)
    
    return config_dir

@pytest.fixture
def config_loader(mock_config_dir, monkeypatch):
    """Create ConfigLoader instance with mocked config directory"""
    monkeypatch.setenv('CONFIG_DIR', str(mock_config_dir))
    return ConfigLoader()

class TestConfigLoader:
    def test_initialization(self, config_loader):
        """Test ConfigLoader initialization"""
        assert config_loader is not None
        assert hasattr(config_loader, 'config_dir')
        assert hasattr(config_loader, 'logger')
    
    def test_load_all_configs(self, config_loader):
        """Test loading all configuration files"""
        for config_name in ConfigLoader.CONFIG_FILES:
            config_name = config_name.replace('.yaml', '')
            config = config_loader.load_config(config_name)
            assert config is not None
    
    def test_get_indicator_config(self, config_loader):
        """Test getting indicator configuration"""
        rsi_config = config_loader.get_indicator_config("rsi")
        assert rsi_config is not None
        assert rsi_config["name"] == "Relative Strength Index"
        assert rsi_config["period"]["default"] == 14
    
    def test_get_strategy_config(self, config_loader):
        """Test getting strategy configuration"""
        strategy_config = config_loader.get_strategy_config("trend_momentum")
        assert strategy_config is not None
        assert strategy_config["name"] == "Trend Momentum Strategy"
        assert "rsi" in strategy_config["indicators"]
    
    def test_get_genetic_config(self, config_loader):
        """Test getting genetic optimization configuration"""
        genetic_config = config_loader.get_genetic_config()
        assert genetic_config is not None
        assert genetic_config["population_size"] == 50
        assert genetic_config["generations"] == 100
    
    def test_get_backtest_config(self, config_loader):
        """Test getting backtest configuration"""
        backtest_config = config_loader.get_backtest_config()
        assert backtest_config is not None
        assert backtest_config["initial_capital"] == 100000
        assert backtest_config["currency"] == "USD"
    
    def test_get_risk_params(self, config_loader):
        """Test getting risk parameters"""
        risk_params = config_loader.get_risk_params()
        assert risk_params is not None
        assert risk_params["risk_free_rate"] == 0.02
        assert risk_params["transaction_costs"] == 0.001
    
    def test_validate_indicator_params(self, config_loader):
        """Test parameter validation for indicators"""
        # Test valid parameters
        valid_params = {"period": {"default": 10}}
        assert config_loader.validate_indicator_params("rsi", valid_params)
        
        # Test parameters out of range
        invalid_params = {"period": {"default": 1}}
        with pytest.raises(ValueError):
            config_loader.validate_indicator_params("rsi", invalid_params)
        
        # Test unknown parameter
        unknown_params = {"unknown": 10}
        with pytest.raises(ValueError):
            config_loader.validate_indicator_params("rsi", unknown_params)
    
    def test_get_all_strategies(self, config_loader):
        """Test getting list of all strategies"""
        strategies = config_loader.get_all_strategies()
        assert isinstance(strategies, list)
        assert "general" in strategies
        assert "trend_momentum" in strategies
    
    def test_get_all_indicators(self, config_loader):
        """Test getting list of all indicators"""
        indicators = config_loader.get_all_indicators()
        assert isinstance(indicators, list)
        assert "rsi" in indicators
    
    def test_update_config_value(self, config_loader):
        """Test updating configuration values"""
        # Test temporary update
        config_loader.update_config_value(
            "indicators",
            "indicators.rsi.period.default",
            20,
            temporary=True
        )
        
        updated_config = config_loader.get_indicator_config("rsi")
        assert updated_config["period"]["default"] == 20
        
        # Test permanent update
        config_loader.update_config_value(
            "indicators",
            "indicators.rsi.period.default",
            25,
            temporary=False
        )
        
        # Reload config to verify persistence
        config_loader.clear_cache()
        reloaded_config = config_loader.get_indicator_config("rsi")
        assert reloaded_config["period"]["default"] == 25
    
    def test_cache_functionality(self, config_loader):
        """Test configuration caching"""
        # First load
        config1 = config_loader.load_config("indicators")
        
        # Second load should return cached version
        config2 = config_loader.load_config("indicators")
        assert config1 is config2
        
        # Clear cache
        config_loader.clear_cache()
        config3 = config_loader.load_config("indicators")
        assert config1 is not config3
    
    def test_missing_config_file(self, mock_config_dir, monkeypatch):
        """Test handling of missing configuration files"""
        # Set the config directory before creating ConfigLoader
        monkeypatch.setenv('CONFIG_DIR', str(mock_config_dir))
        
        # Remove a config file
        os.remove(mock_config_dir / "indicators.yaml")
        
        # Create a new ConfigLoader - should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ConfigLoader()
    
    def test_invalid_yaml(self, config_loader, mock_config_dir):
        """Test handling of invalid YAML files"""
        # Create invalid YAML
        with open(mock_config_dir / "indicators.yaml", 'w') as f:
            f.write("invalid: yaml: {unclosed")
        
        with pytest.raises(ValueError):
            config_loader.load_config("indicators")
