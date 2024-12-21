from typing import Any, Tuple
import pandas as pd
import talib
from deap import tools
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class SMAGene(BaseGene):
    """Gene class for Simple Moving Average indicator using TA-Lib and DEAP"""
    
    def __init__(self):
        """Initialize SMA gene with configuration from config file"""
        config = ConfigLoader.get_indicator_config("sma")
        gene_config = GeneConfig(
            name=config["name"],
            min_value=config["min_period"],
            max_value=config["max_period"],
            step=config["step"],
            mutation_rate=config["mutation_rate"],
            mutation_sigma=config["mutation_range"]
        )
        super().__init__(gene_config)
        
        # Override mutation operator for integer values
        self.toolbox.register("mutate", tools.mutGaussian, 
                            mu=0, sigma=self.config.mutation_sigma, 
                            indpb=self.config.mutation_rate)
        
        self.randomize()
        
        # Cache for computed values to avoid recalculation
        self._cache = {}
    
    def compute(self, data: pd.Series) -> pd.Series:
        """
        Compute Simple Moving Average for the given data using TA-Lib
        
        Args:
            data (pd.Series): Price data series
            
        Returns:
            pd.Series: Computed SMA values
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series")
        
        # Check cache first
        cache_key = (data.name, self.value)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        period = int(self.value)
        result = pd.Series(talib.SMA(data, timeperiod=period), index=data.index)
        
        # Cache the result
        self._cache[cache_key] = result
        return result
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of an individual (SMA period)
        This method should be extended based on specific strategy requirements
        
        Returns:
            tuple: Fitness score as a tuple (required by DEAP)
        """
        period = int(individual[0])
        
        # Basic fitness function - can be customized based on strategy
        # Penalize very short and very long periods
        normalized_period = (period - self.config.min_value) / (self.config.max_value - self.config.min_value)
        fitness = 1.0 - abs(0.5 - normalized_period)  # Prefer middle-range periods
        
        return (fitness,)
    
    def validate_and_clip_value(self, value: float) -> float:
        """
        Validate and clip the value to be within the allowed range and ensure it's an integer
        
        Args:
            value (float): Value to validate and clip
            
        Returns:
            float: Validated and clipped value
        """
        clipped = super().validate_and_clip_value(value)
        return round(clipped)  # Round to nearest integer
    
    def crossover(self, other: 'SMAGene') -> tuple['SMAGene', 'SMAGene']:
        """
        Perform crossover with another SMA gene, ensuring integer results
        """
        child1, child2 = super().crossover(other)
        
        # Ensure integer values
        child1.value = round(child1.value)
        child2.value = round(child2.value)
        
        return child1, child2
    
    def to_dict(self) -> dict:
        """
        Convert the gene to a dictionary representation with SMA-specific information
        
        Returns:
            dict: Dictionary representation of the SMA gene
        """
        base_dict = super().to_dict()
        base_dict.update({
            "period": int(self.value),
            "indicator_type": "SMA",
            "library": "TA-Lib"
        })
        return base_dict