from typing import Any, Tuple
import pandas as pd
import talib
from deap import tools
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class ATRGene(BaseGene):
    """Gene class for Average True Range indicator using TA-Lib and DEAP"""
    
    def __init__(self):
        """Initialize ATR gene with configuration"""
        config = ConfigLoader.get_indicator_config("atr")
        gene_config = GeneConfig(
            name=config["name"],
            min_value=config["min_period"],
            max_value=config["max_period"],
            step=config["step"],
            mutation_rate=config["mutation_rate"],
            mutation_sigma=config["mutation_range"]
        )
        super().__init__(gene_config)
        self.toolbox.register("mutate", tools.mutGaussian, 
                            mu=0, sigma=self.config.mutation_sigma, 
                            indpb=self.config.mutation_rate)
        self.randomize()
        self._cache = {}
    
    def compute(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Compute ATR for the given data using TA-Lib
        
        Args:
            high (pd.Series): High price data series
            low (pd.Series): Low price data series
            close (pd.Series): Close price data series
            
        Returns:
            pd.Series: Computed ATR values
        """
        if not all(isinstance(x, pd.Series) for x in [high, low, close]):
            raise ValueError("All inputs must be pandas Series")
        
        cache_key = (high.name, self.value)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        period = int(self.value)
        result = pd.Series(
            talib.ATR(high, low, close, timeperiod=period),
            index=high.index
        )
        
        self._cache[cache_key] = result
        return result
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of an individual (ATR period)
        Prefers standard period of 14 days
        """
        period = int(individual[0])
        
        # Optimal ATR period
        optimal_period = 14
        
        # Calculate distance from optimal period
        distance = abs(period - optimal_period)
        fitness = 1.0 / (1.0 + distance)
        
        return (fitness,)
    
    def validate_and_clip_value(self, value: float) -> float:
        """Ensure integer periods"""
        clipped = super().validate_and_clip_value(value)
        return round(clipped)
    
    def to_dict(self) -> dict:
        """Dictionary representation with ATR-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "period": int(self.value),
            "indicator_type": "ATR",
            "library": "TA-Lib",
            "description": "Volatility indicator measuring average range"
        })
        return base_dict