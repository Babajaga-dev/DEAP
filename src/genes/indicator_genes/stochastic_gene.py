from typing import Any, Tuple
import pandas as pd
import talib
import random
from deap import tools, creator, base
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class StochasticGene(BaseGene):
    """Gene class for Stochastic Oscillator using TA-Lib and DEAP"""
    
    def __init__(self, config_loader: ConfigLoader = None):
        """
        Initialize Stochastic gene with configuration
        
        Args:
            config_loader (ConfigLoader, optional): Configuration loader instance
        """
        if config_loader is None:
            config_loader = ConfigLoader()
            
        self._config_loader = config_loader  # Store for crossover operations
        config = config_loader.get_indicator_config("stochastic")
        
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
        
        gene_type_name = f"{self.__class__.__name__}Class"
        self.toolbox.register("individual", tools.initCycle, 
                            getattr(creator, gene_type_name),
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
        
        # Validate fastk_period (5-30)
        if new_value[0] < 5:
            raise ValueError("fastk_period must be at least 5")
        self.fastk_period = int(self.validate_and_clip_value(new_value[0]))
        
        # Validate slowk_period (1-10)
        if new_value[1] < 1:
            raise ValueError("slowk_period must be at least 1")
        self.slowk_period = min(10, int(new_value[1]))
        
        # Validate slowd_period (1-10)
        if new_value[2] < 1:
            raise ValueError("slowd_period must be at least 1")
        self.slowd_period = min(10, int(new_value[2]))
    
    def crossover(self, other: 'StochasticGene') -> tuple['StochasticGene', 'StochasticGene']:
        """Perform crossover with another Stochastic gene"""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot crossover with gene of different type: {type(other)}")
        
        # Create new instances with same config loader
        child1 = self.__class__(self._config_loader)
        child2 = self.__class__(self._config_loader)
        
        # Perform arithmetic crossover for each parameter
        alpha = random.random()
        child1.value = [
            round(alpha * self.fastk_period + (1 - alpha) * other.fastk_period),
            round(alpha * self.slowk_period + (1 - alpha) * other.slowk_period),
            round(alpha * self.slowd_period + (1 - alpha) * other.slowd_period)
        ]
        child2.value = [
            round((1 - alpha) * self.fastk_period + alpha * other.fastk_period),
            round((1 - alpha) * self.slowk_period + alpha * other.slowk_period),
            round((1 - alpha) * self.slowd_period + alpha * other.slowd_period)
        ]
        
        return child1, child2

    def mutate(self) -> None:
        """Perform mutation on all three parameters"""
        # Mutate fastk_period
        if random.random() < self.config.mutation_rate:
            self.fastk_period = max(5, min(30, round(
                self.fastk_period + random.gauss(0, self.config.mutation_sigma * 5)
            )))
        
        # Mutate slowk_period
        if random.random() < self.config.mutation_rate:
            self.slowk_period = max(1, min(10, round(
                self.slowk_period + random.gauss(0, self.config.mutation_sigma * 2)
            )))
        
        # Mutate slowd_period
        if random.random() < self.config.mutation_rate:
            self.slowd_period = max(1, min(10, round(
                self.slowd_period + random.gauss(0, self.config.mutation_sigma * 2)
            )))

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
