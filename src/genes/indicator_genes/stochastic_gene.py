from typing import Any, Tuple
import pandas as pd
import talib
from deap import tools, creator, base
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class StochasticGene(BaseGene):
    """Gene class for Stochastic Oscillator using TA-Lib and DEAP"""
    
    def __init__(self):
        """Initialize Stochastic gene with configuration"""
        config = ConfigLoader.get_indicator_config("stochastic")
        
        # Multiple parameters for Stochastic
        self.fastk_period = config["fastk_period"]["default"]  # Default 14
        self.slowk_period = config["slowk_period"]["default"]  # Default 3
        self.slowd_period = config["slowd_period"]["default"]  # Default 3
        
        gene_config = GeneConfig(
            name=config["name"],
            min_value=config["fastk_period"]["min"],
            max_value=config["fastk_period"]["max"],
            step=config["step"],
            mutation_rate=config["mutation_rate"],
            mutation_sigma=config["mutation_range"]
        )
        super().__init__(gene_config)
        self._register_stochastic_operators()
        self.randomize()
        self._cache = {}
    
    def _register_stochastic_operators(self):
        """Register Stochastic-specific genetic operators"""
        self.toolbox.register("attr_fastk", random.randint, 
                            self.config.min_value, 
                            self.config.max_value)
        self.toolbox.register("attr_slowk", random.randint, 1, 10)
        self.toolbox.register("attr_slowd", random.randint, 1, 10)
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.attr_fastk, self.toolbox.attr_slowk, 
                             self.toolbox.attr_slowd), n=1)
    
    def compute(self, data: pd.Series, high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Compute Stochastic Oscillator for the given data using TA-Lib
        
        Args:
            data (pd.Series): Close price data series
            high (pd.Series): High price data series
            low (pd.Series): Low price data series
            
        Returns:
            tuple: (Slow %K, Slow %D)
        """
        if not all(isinstance(x, pd.Series) for x in [data, high, low]):
            raise ValueError("All inputs must be pandas Series")
        
        cache_key = (data.name, self.fastk_period, self.slowk_period, self.slowd_period)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        slowk, slowd = talib.STOCH(
            high, low, data,
            fastk_period=int(self.fastk_period),
            slowk_period=int(self.slowk_period),
            slowk_matype=0,
            slowd_period=int(self.slowd_period),
            slowd_matype=0
        )
        
        result = (
            pd.Series(slowk, index=data.index),
            pd.Series(slowd, index=data.index)
        )
        
        self._cache[cache_key] = result
        return result
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of Stochastic parameters
        Prefers standard parameters (14, 3, 3)
        """
        fastk, slowk, slowd = individual
        
        # Optimal parameters
        optimal_fastk = 14
        optimal_slowk = 3
        optimal_slowd = 3
        
        # Calculate distances from optimal parameters
        fastk_diff = abs(fastk - optimal_fastk) / optimal_fastk
        slowk_diff = abs(slowk - optimal_slowk) / optimal_slowk
        slowd_diff = abs(slowd - optimal_slowd) / optimal_slowd
        
        # Combined fitness score
        fitness = 1.0 / (1.0 + fastk_diff + slowk_diff + slowd_diff)
        
        return (fitness,)
    
    @property
    def value(self) -> tuple[int, int, int]:
        """Get the current Stochastic parameters"""
        return (self.fastk_period, self.slowk_period, self.slowd_period)
    
    @value.setter
    def value(self, new_value: list) -> None:
        """Set Stochastic parameters with validation"""
        if len(new_value) != 3:
            raise ValueError("Stochastic requires three parameters")
        
        self.fastk_period = int(self.validate_and_clip_value(new_value[0]))
        self.slowk_period = max(1, min(10, int(new_value[1])))
        self.slowd_period = max(1, min(10, int(new_value[2])))
    
    def to_dict(self) -> dict:
        """Dictionary representation with Stochastic-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "fastk_period": int(self.fastk_period),
            "slowk_period": int(self.slowk_period),
            "slowd_period": int(self.slowd_period),
            "indicator_type": "Stochastic",
            "library": "TA-Lib",
            "range": "0-100"
        })
        return base_dict