from typing import Dict, Any, Optional, List
import os
import yaml
from functools import lru_cache
import logging
from pathlib import Path

class ConfigLoader:
    """Enhanced configuration loader for all system components"""
    
    def __init__(self, config_dir: Optional[str | Path] = None):
        """
        Initialize ConfigLoader
        
        Args:
            config_dir: Optional path to config directory. If not provided,
                       will use CONFIG_DIR env var or default to config/
        """
        self.config_dir = str(config_dir or os.getenv('CONFIG_DIR') or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'config'
        ))
        self.setup_logging()
        self._cached_configs = {}

    def setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _validate_config_file(self, filename: str) -> None:
        """Validate existence of a specific configuration file"""
        file_path = os.path.join(self.config_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {filename}")

    @lru_cache(maxsize=32)
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a specific configuration file"""
        filename = f"{config_name}.yaml"
        self._validate_config_file(filename)
        config_path = os.path.join(self.config_dir, filename)
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                if config is None:
                    config = {}
                self._cached_configs[config_name] = config
                self.logger.info(f"Successfully loaded configuration: {config_name}")
                return config
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file {config_name}: {e}")
            raise ValueError(f"Error parsing configuration file: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration {config_name}: {e}")
            raise

    def get_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        """Get configuration for a specific indicator"""
        config = self.load_config("indicators")
        if "indicators" not in config or indicator_name not in config["indicators"]:
            raise ValueError(f"Indicator {indicator_name} not found in configuration")
        return config["indicators"][indicator_name]

    def get_strategy_config(self, strategy_name: str = None) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        config = self.load_config("strategies")
        if strategy_name:
            if "strategies" not in config or strategy_name not in config["strategies"]:
                raise ValueError(f"Strategy {strategy_name} not found in configuration")
            
            # Create a deep copy of the strategy configuration
            import copy
            strategy_config = copy.deepcopy(config["strategies"][strategy_name])
            
            # Ensure all required sections exist
            if "position_sizing" not in strategy_config:
                strategy_config["position_sizing"] = {"method": "fixed", "base_size": 1.0}
            if "entry_conditions" not in strategy_config:
                strategy_config["entry_conditions"] = {}
            if "exit_conditions" not in strategy_config:
                strategy_config["exit_conditions"] = {}
            if "risk_management" not in strategy_config:
                strategy_config["risk_management"] = {}
                
            return strategy_config
        return config

    def get_genetic_config(self) -> Dict[str, Any]:
        """Get genetic optimization configuration"""
        config = self.load_config("genetic")
        return config.get("optimization", {})

    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtesting configuration"""
        config = self.load_config("backtest")
        return config.get("backtest", {})
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.load_config("data")

    def get_indicators_config(self) -> Dict[str, Any]:
        """Get indicators configuration"""
        return self.load_config("indicators")

    def get_risk_params(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        config = self.load_config("strategies")
        return config["strategies"]["general"]

    def validate_indicator_params(self, indicator_name: str, params: Dict[str, Any]) -> bool:
        """Validate parameters for a specific indicator"""
        config = self.get_indicator_config(indicator_name)
        
        for param_name, param_config_value in params.items():
            if param_name not in config:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            param_config = config[param_name]
            if isinstance(param_config, dict) and "min" in param_config and "max" in param_config:
                # Estrai il valore effettivo dal dizionario dei parametri
                param_value = param_config_value.get("default") if isinstance(param_config_value, dict) else param_config_value
                
                if param_value < param_config["min"]:
                    raise ValueError(
                        f"Parameter {param_name} value {param_value} is below minimum {param_config['min']}"
                    )
                if param_value > param_config["max"]:
                    raise ValueError(
                        f"Parameter {param_name} value {param_value} exceeds maximum {param_config['max']}"
                    )
        
        return True

    def get_all_strategies(self) -> List[str]:
        """Get list of all available strategies"""
        config = self.load_config("strategies")
        return list(config["strategies"].keys())

    def get_all_indicators(self) -> List[str]:
        """Get list of all available indicators"""
        config = self.load_config("indicators")
        return list(config["indicators"].keys())

    def update_config_value(self, config_name: str, path: str, value: Any, temporary: bool = True) -> None:
        """Update a configuration value"""
        config = self.load_config(config_name)
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value
        
        if not temporary:
            config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
            with open(config_path, 'w') as file:
                yaml.safe_dump(config, file)
        
        self._cached_configs[config_name] = config
        self.logger.info(f"Updated configuration value: {path}")

    def clear_cache(self) -> None:
        """Clear the configuration cache"""
        self._cached_configs.clear()
        self.load_config.cache_clear()
        self.logger.info("Configuration cache cleared")
