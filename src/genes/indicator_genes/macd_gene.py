from typing import Any, Tuple
import pandas as pd
import talib
import random
from deap import tools, creator, base
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class MACDGene(BaseGene):
    """Gene class for MACD indicator using TA-Lib and DEAP"""
    
    def __init__(self):
        """Initialize MACD gene with configuration from config file"""
        config_loader = ConfigLoader()
        config = config_loader.get_indicator_config("macd")
        
        # MACD needs three parameters: fast period, slow period, and signal period
        self.fast_period = config["fast_period"]["default"]
        self.slow_period = config["slow_period"]["default"]
        self.signal_period = config["signal_period"]["default"]
        
        # Use the slow period range for the gene config
        gene_config = GeneConfig(
            name=config["name"],
            min_value=config["fast_period"]["min"],
            max_value=config["slow_period"]["max"],
            step=config["step"],
            mutation_rate=config["mutation_rate"],
            mutation_sigma=config["mutation_range"]
        )
        super().__init__(gene_config)
        
        # Override genetic operators for multiple parameters
        self._register_macd_operators()
        self.randomize()
        self._cache = {}
    
    def _register_macd_operators(self):
        """Register MACD-specific genetic operators"""
        # Create individual with three parameters
        self.toolbox.register("attr_fast", random.randint, 
                            self.config.min_value, 
                            self.config.max_value)
        self.toolbox.register("attr_slow", random.randint, 
                            self.config.min_value, 
                            self.config.max_value)
        self.toolbox.register("attr_signal", random.randint, 
                            self.config.min_value, 
                            self.config.max_value)
        
        gene_type_name = f"{self.__class__.__name__}Class"
        self.toolbox.register("individual", tools.initCycle, 
                            getattr(creator, gene_type_name),
                            (self.toolbox.attr_fast, self.toolbox.attr_slow, 
                             self.toolbox.attr_signal), n=1)
    
    def compute(self, data: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MACD for the given data using TA-Lib
        
        Args:
            data (pd.Series): Price data series
            
        Returns:
            tuple: (MACD line, Signal line, MACD histogram)
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series")
        
        cache_key = (data.name, self.fast_period, self.slow_period, self.signal_period)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        macd, signal, hist = talib.MACD(
            data,
            fastperiod=int(self.fast_period),
            slowperiod=int(self.slow_period),
            signalperiod=int(self.signal_period)
        )
        
        result = (
            pd.Series(macd, index=data.index),
            pd.Series(signal, index=data.index),
            pd.Series(hist, index=data.index)
        )
        
        self._cache[cache_key] = result
        return result
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of MACD parameters
        Ensures fast period < slow period and reasonable signal period
        """
        fast, slow, signal = individual
        
        if fast >= slow:  # Fast period should be smaller than slow period
            return (-1.0,)
        
        # Prefer traditional MACD parameters (12, 26, 9)
        optimal_fast, optimal_slow, optimal_signal = 12, 26, 9
        
        # Calculate distance from optimal parameters
        fast_diff = abs(fast - optimal_fast)
        slow_diff = abs(slow - optimal_slow)
        signal_diff = abs(signal - optimal_signal)
        
        # Combined fitness score
        fitness = 1.0 / (1.0 + fast_diff + slow_diff + signal_diff)
        
        return (fitness,)
    
    @property
    def value(self) -> list:
        """Get the current MACD parameters"""
        return [self.fast_period, self.slow_period, self.signal_period]
    
    @value.setter
    def value(self, new_value: list) -> None:
        """Set MACD parameters with validation"""
        if len(new_value) != 3:
            raise ValueError("MACD requires three parameters")
        
        fast, slow, signal = map(self.validate_and_clip_value, new_value)
        
        # Ensure fast period is smaller than slow period
        if fast >= slow:
            fast, slow = slow - 1, slow
        
        self._value = [fast, slow, signal]
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
    
    def mutate(self) -> None:
        """Mutate the MACD parameters"""
        # Perform mutation using DEAP's operator
        mutated_value, = self.toolbox.mutate(self._value)
        
        # Update the values ensuring they are valid
        self.value = mutated_value
    
    def crossover(self, other: 'MACDGene') -> tuple['MACDGene', 'MACDGene']:
        """Perform crossover with another MACD gene"""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot crossover with gene of different type: {type(other)}")
        
        # Create new genes with same configuration
        child1, child2 = MACDGene(), MACDGene()
        
        # Perform crossover using DEAP's operator
        offspring1, offspring2 = self.toolbox.mate(self._value, other._value)
        
        # Set the values for children
        child1.value = offspring1
        child2.value = offspring2
        
        return child1, child2
    
    def to_dict(self) -> dict:
        """Dictionary representation with MACD-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "fast_period": int(self.fast_period),
            "slow_period": int(self.slow_period),
            "signal_period": int(self.signal_period),
            "indicator_type": "MACD",
            "library": "TA-Lib"
        })
        return base_dict
