from typing import Any, Tuple
import pandas as pd
import talib
from deap import tools
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class EMAGene(BaseGene):
    """Gene class for Exponential Moving Average indicator using TA-Lib and DEAP"""
    
    def __init__(self):
        """Initialize EMA gene with configuration"""
        config = ConfigLoader.get_indicator_config("ema")
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
    
    def compute(self, data: pd.Series) -> pd.Series:
        """
        Compute EMA for the given data using TA-Lib
        
        Args:
            data (pd.Series): Price data series
            
        Returns:
            pd.Series: Computed EMA values
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series")
        
        cache_key = (data.name, self.value)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        period = int(self.value)
        result = pd.Series(
            talib.EMA(data, timeperiod=period),
            index=data.index
        )
        
        self._cache[cache_key] = result
        return result
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of an individual (EMA period)
        Prefers common EMA periods (e.g., 9, 21, 50, 200)
        """
        period = int(individual[0])
        
        # Common EMA periods
        common_periods = [9, 21, 50, 200]
        
        # Find closest common period
        min_distance = min(abs(period - cp) for cp in common_periods)
        
        # Fitness based on proximity to common periods
        fitness = 1.0 / (1.0 + min_distance)
        
        return (fitness,)
    
    def validate_and_clip_value(self, value: float) -> float:
        """Ensure integer periods"""
        clipped = super().validate_and_clip_value(value)
        return round(clipped)
    
    def crossover(self, other: 'EMAGene') -> tuple['EMAGene', 'EMAGene']:
        """Custom crossover that maintains period hierarchy if needed"""
        child1, child2 = super().crossover(other)
        return child1, child2
    
    def to_dict(self) -> dict:
        """Dictionary representation with EMA-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "period": int(self.value),
            "indicator_type": "EMA",
            "library": "TA-Lib",
            "description": "Exponential Moving Average"
        })
        return base_dict
    
    @staticmethod
    def calculate_multiplier(period: int) -> float:
        """Calculate EMA multiplier for given period"""
        return 2.0 / (period + 1)