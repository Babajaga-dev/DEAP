import pytest
import os
import yaml
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def config_loader():
    """Fixture that provides a ConfigLoader instance"""
    return ConfigLoader()

@pytest.fixture
def mock_config_data():
    """Fixture that provides test configuration data"""
    return {
        "indicators": {
            "sma": {
                "name": "Simple Moving Average",
                "period": {
                    "min": 2,
                    "max": 200,
                    "default": 20
                },
                "mutation_rate": {
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.1
                },
                "mutation_range": {
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.2
                }
            },
            "rsi": {
                "name": "Relative Strength Index",
                "period": {
                    "min": 2,
                    "max": 50,
                    "default": 14
                },
                "overbought": {
                    "min": 50,
                    "max": 100,
                    "default": 70
                },
                "oversold": {
                    "min": 0,
                    "max": 50,
                    "default": 30
                }
            }
        },
        "general": {
            "optimization": {
                "population_size": 50,
                "generations": 100,
                "crossover_probability": 0.7,
                "mutation_probability": 0.2
            },
            "risk_management": {
                "max_position_size": 0.1,
                "max_risk_per_trade": 0.02,
                "stop_loss_atr_multiplier": 2.0
            }
        }
    }

@pytest.fixture
def mock_config_file(tmp_path, mock_config_data):
    """Fixture that creates a temporary config file with test data"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "indicators.yaml"
    
    with open(config_file, 'w') as f:
        yaml.dump(mock_config_data, f)
    
    return config_file

@pytest.fixture
def invalid_yaml_file(tmp_path):
    """Fixture that creates an invalid YAML file"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "invalid.yaml"
    
    with open(config_file, 'w') as f:
        f.write("invalid: yaml: {unclosed: bracket")
    
    return config_file

class TestConfigLoader:
    def test_initialization(self, config_loader):
        """Test ConfigLoader initialization"""
        assert config_loader is not None
        assert hasattr(config_loader, 'config_dir')
        assert hasattr(config_loader, 'logger')
        assert isinstance(config_loader._cached_configs, dict)

    def test_load_config(self, config_loader, mock_config_file, monkeypatch):
        """Test loading configuration file"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        config = config_loader.load_config("indicators")
        
        assert config is not None
        assert "indicators" in config
        assert "general" in config
        assert "sma" in config["indicators"]
        assert "rsi" in config["indicators"]

    def test_load_nonexistent_config(self, config_loader):
        """Test loading non-existent configuration file"""
        with pytest.raises(FileNotFoundError) as exc_info:
            config_loader.load_config("nonexistent")
        assert "not found" in str(exc_info.value)

    def test_load_invalid_yaml(self, config_loader, invalid_yaml_file, monkeypatch):
        """Test loading invalid YAML file"""
        monkeypatch.setattr(config_loader, 'config_dir', str(invalid_yaml_file.parent))
        with pytest.raises(ValueError) as exc_info:
            config_loader.load_config("invalid")
        assert "Error parsing configuration file" in str(exc_info.value)

    def test_get_indicator_config(self, config_loader, mock_config_file, monkeypatch):
        """Test getting specific indicator configuration"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        sma_config = config_loader.get_indicator_config("sma")
        
        assert sma_config is not None
        assert sma_config["name"] == "Simple Moving Average"
        assert sma_config["period"]["min"] == 2
        assert sma_config["period"]["max"] == 200
        assert sma_config["period"]["default"] == 20

    def test_get_nonexistent_indicator(self, config_loader, mock_config_file, monkeypatch):
        """Test getting non-existent indicator configuration"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        with pytest.raises(ValueError) as exc_info:
            config_loader.get_indicator_config("nonexistent")
        assert "not found" in str(exc_info.value)

    def test_validate_indicator_params(self, config_loader, mock_config_file, monkeypatch):
        """Test parameter validation for indicators"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        
        # Test valid parameters
        valid_params = {"period": 20, "mutation_rate": 0.1}
        assert config_loader.validate_indicator_params("sma", valid_params)
        
        # Test parameter below minimum
        with pytest.raises(ValueError) as exc_info:
            config_loader.validate_indicator_params("sma", {"period": 1})
        assert "below minimum" in str(exc_info.value)
        
        # Test parameter above maximum
        with pytest.raises(ValueError) as exc_info:
            config_loader.validate_indicator_params("sma", {"period": 201})
        assert "exceeds maximum" in str(exc_info.value)
        
        # Test unknown parameter
        with pytest.raises(ValueError) as exc_info:
            config_loader.validate_indicator_params("sma", {"unknown": 10})
        assert "Unknown parameter" in str(exc_info.value)

    def test_validate_rsi_params(self, config_loader, mock_config_file, monkeypatch):
        """Test parameter validation for RSI indicator"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        
        # Test valid parameters
        valid_params = {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        }
        assert config_loader.validate_indicator_params("rsi", valid_params)
        
        # Test invalid overbought level
        with pytest.raises(ValueError) as exc_info:
            config_loader.validate_indicator_params("rsi", {"overbought": 101})
        assert "exceeds maximum" in str(exc_info.value)
        
        # Test invalid oversold level
        with pytest.raises(ValueError) as exc_info:
            config_loader.validate_indicator_params("rsi", {"oversold": -1})
        assert "below minimum" in str(exc_info.value)

    def test_get_optimization_params(self, config_loader, mock_config_file, monkeypatch):
        """Test getting optimization parameters"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        opt_params = config_loader.get_optimization_params()
        
        assert opt_params is not None
        assert opt_params["population_size"] == 50
        assert opt_params["generations"] == 100
        assert opt_params["crossover_probability"] == 0.7
        assert opt_params["mutation_probability"] == 0.2

    def test_get_risk_params(self, config_loader, mock_config_file, monkeypatch):
        """Test getting risk management parameters"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        risk_params = config_loader.get_risk_params()
        
        assert risk_params is not None
        assert risk_params["max_position_size"] == 0.1
        assert risk_params["max_risk_per_trade"] == 0.02
        assert risk_params["stop_loss_atr_multiplier"] == 2.0

    def test_update_config_value(self, config_loader, mock_config_file, monkeypatch):
        """Test updating configuration values"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        
        # Test temporary update
        config_loader.update_config_value(
            "indicators",
            "indicators.sma.period.default",
            30,
            temporary=True
        )
        
        updated_config = config_loader.get_indicator_config("sma")
        assert updated_config["period"]["default"] == 30
        
        # Verify file wasn't modified
        with open(mock_config_file, 'r') as f:
            file_config = yaml.safe_load(f)
        assert file_config["indicators"]["sma"]["period"]["default"] == 20
        
        # Test persistent update
        config_loader.update_config_value(
            "indicators",
            "indicators.sma.period.default",
            40,
            temporary=False
        )
        
        # Verify file was modified
        with open(mock_config_file, 'r') as f:
            file_config = yaml.safe_load(f)
        assert file_config["indicators"]["sma"]["period"]["default"] == 40

    def test_cache_functionality(self, config_loader, mock_config_file, monkeypatch):
        """Test configuration caching"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        
        # First load
        config1 = config_loader.load_config("indicators")
        
        # Second load should return cached version
        config2 = config_loader.load_config("indicators")
        
        assert config1 is config2  # Same object due to caching
        
        # Clear cache
        config_loader.clear_cache()
        config3 = config_loader.load_config("indicators")
        
        assert config1 is not config3  # Different object after cache clear

    def test_nested_config_update(self, config_loader, mock_config_file, monkeypatch):
        """Test updating deeply nested configuration values"""
        monkeypatch.setattr(config_loader, 'config_dir', str(mock_config_file.parent))
        
        path = "general.optimization.population_size"
        config_loader.update_config_value("indicators", path, 100, temporary=True)
        
        opt_params = config_loader.get_optimization_params()
        assert opt_params["population_size"] == 100