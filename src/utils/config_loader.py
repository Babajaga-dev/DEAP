from typing import Dict, Any, Optional, Union
import os
import yaml
from functools import lru_cache
import logging

class ConfigLoader:
    def __init__(self):
        self.config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'config'
        )
        self.setup_logging()
        self._cached_configs = {}

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=32)
    def load_config(self, config_name: str) -> Dict[str, Any]:
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
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
        config = self.load_config("indicators")
        if "indicators" not in config or indicator_name not in config["indicators"]:
            self.logger.error(f"Indicator {indicator_name} not found in configuration")
            raise ValueError(f"Indicator {indicator_name} not found in configuration")
        return config["indicators"][indicator_name]

    def validate_indicator_params(self, indicator_name: str, params: Dict[str, Any]) -> bool:
        """
        Validate parameters for a specific indicator
        
        Args:
            indicator_name (str): Name of the indicator
            params (Dict[str, Any]): Parameters to validate
            
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        indicator_config = self.get_indicator_config(indicator_name)

        for param_name, param_value in params.items():
            # Verifica se il parametro esiste nella configurazione
            if param_name not in indicator_config:
                raise ValueError(f"Unknown parameter: {param_name}")

            param_config = indicator_config[param_name]

            # Se il parametro ha una configurazione dettagliata (dizionario con min/max)
            if isinstance(param_config, dict) and "min" in param_config and "max" in param_config:
                if param_value < param_config["min"]:
                    raise ValueError(
                        f"Parameter {param_name} value {param_value} is below minimum {param_config['min']}"
                    )
                if param_value > param_config["max"]:
                    raise ValueError(
                        f"Parameter {param_name} value {param_value} exceeds maximum {param_config['max']}"
                    )

        return True

    def get_general_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        config = self.load_config("indicators")
        general_config = config.get("general", {})
        
        if section:
            return general_config.get(section, {})
        return general_config

    def get_optimization_params(self) -> Dict[str, Any]:
        return self.get_general_config("optimization")

    def get_risk_params(self) -> Dict[str, Any]:
        return self.get_general_config("risk_management")

    def update_config_value(self, config_name: str, path: str, value: Any, temporary: bool = True) -> None:
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
        self._cached_configs.clear()
        self.load_config.cache_clear()
        self.logger.info("Configuration cache cleared")