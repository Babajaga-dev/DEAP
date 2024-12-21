from typing import Any, Tuple
import pandas as pd
import numpy as np
import talib
from deap import tools
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class RSIGene(BaseGene):
    """Gene class for Relative Strength Index indicator using TA-Lib and DEAP"""
    
    def __init__(self, config_loader: ConfigLoader = None):
        """
        Initialize RSI gene with configuration
        
        Args:
            config_loader (ConfigLoader, optional): Configuration loader instance
        """
        if config_loader is None:
            config_loader = ConfigLoader()
            
        self._config_loader = config_loader  # Store for crossover operations
        config = config_loader.get_indicator_config("rsi")
        
        # Ensure we have default values if configuration is incomplete
        min_period = 2
        max_period = 50
        if "period" in config and isinstance(config["period"], dict):
            min_period = config["period"].get("min", min_period)
            max_period = config["period"].get("max", max_period)
        else:
            min_period = config.get("min_period", min_period)
            max_period = config.get("max_period", max_period)
        
        gene_config = GeneConfig(
            name=config.get("name", "RSI"),
            min_value=float(min_period),
            max_value=float(max_period),
            step=float(config.get("step", 1)),
            mutation_rate=config.get("mutation_rate", 0.1),
            mutation_sigma=config.get("mutation_range", 0.2)
        )
        
        super().__init__(gene_config)
        
        # Override mutation operator for integer values
        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=max(1, (max_period - min_period) * 0.1),  # Ensure reasonable mutation range
            indpb=1.0  # Ensure mutation happens
        )
        
        self.randomize()
        self._cache = {}

    @property
    def value(self) -> float:
        """Get the current value of the gene"""
        if isinstance(self._value, (list, np.ndarray)):
            return float(round(self._value[0]))
        return float(round(self._value))
    
    @value.setter
    def value(self, new_value: Any) -> None:
        """Set the value of the gene with validation and rounding"""
        if isinstance(new_value, (list, np.ndarray)):
            self._value = [round(self.validate_and_clip_value(new_value[0]))]
        else:
            self._value = round(self.validate_and_clip_value(new_value))
    
    def compute(self, data: pd.Series) -> pd.Series:
        """
        Compute RSI for the given data using TA-Lib
        
        Args:
            data (pd.Series): Price data series
            
        Returns:
            pd.Series: Computed RSI values (0-100)
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series")
        
        cache_key = (data.name, self.value)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        period = int(self.value)
        result = pd.Series(
            talib.RSI(data.values, timeperiod=period),
            index=data.index
        )
        
        self._cache[cache_key] = result
        return result
    
    def validate_and_clip_value(self, value: float) -> float:
        """Ensure integer periods within valid range"""
        clipped = super().validate_and_clip_value(float(value))
        return round(clipped)
    
    def mutate(self) -> None:
        """Perform mutation operation"""
        old_value = self.value
        for _ in range(10):  # Try multiple mutations to ensure change
            self.toolbox.mutate(self._value)
            new_value = self.validate_and_clip_value(self._value[0])
            if new_value != old_value:
                self.value = new_value
                return
    
    def crossover(self, other: 'RSIGene') -> tuple['RSIGene', 'RSIGene']:
        """Perform crossover with another RSI gene"""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot crossover with gene of different type: {type(other)}")
        
        # Create new instances with same config loader
        child1 = self.__class__(self._config_loader)
        child2 = self.__class__(self._config_loader)
        
        # Perform arithmetic crossover
        alpha = np.random.random()
        child1.value = round(alpha * self.value + (1 - alpha) * other.value)
        child2.value = round((1 - alpha) * self.value + alpha * other.value)
        
        return child1, child2
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of an individual (RSI period)
        Prefers periods that are commonly used in trading (around 14)
        """
        period = int(individual[0])
        optimal_period = 14
        distance = abs(period - optimal_period)
        fitness = 1.0 / (1.0 + distance)
        return (fitness,)
    
    def to_dict(self) -> dict:
        """Dictionary representation with RSI-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "period": int(self.value),
            "indicator_type": "RSI",
            "library": "TA-Lib",
            "range": "0-100"
        })
        return base_dict

    def analyze_oversold_overbought(self, data: pd.Series, 
                                  oversold: float = 30, 
                                  overbought: float = 70) -> pd.Series:
        """
        Analyze oversold and overbought conditions
        
        Args:
            data (pd.Series): Price data series
            oversold (float): Oversold threshold (default: 30)
            overbought (float): Overbought threshold (default: 70)
            
        Returns:
            pd.Series: Signal series (-1: oversold, 0: neutral, 1: overbought)
        """
        rsi = self.compute(data)
        signals = pd.Series(0, index=data.index)
        signals[rsi <= oversold] = -1  # Oversold
        signals[rsi >= overbought] = 1  # Overbought
        return signals