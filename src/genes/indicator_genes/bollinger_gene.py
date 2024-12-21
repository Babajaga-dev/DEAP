from typing import Any, Tuple, Optional
import pandas as pd
import talib
import random
from deap import tools, creator, base
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class BollingerGene(BaseGene):
    """Gene class for Bollinger Bands indicator using TA-Lib and DEAP"""
    
    def __init__(self, gene_config: Optional[GeneConfig] = None):
        """Initialize Bollinger Bands gene with configuration"""
        if gene_config is None:
            config_loader = ConfigLoader()
            config = config_loader.get_indicator_config("bollinger")
            
            # We need period and number of standard deviations
            self.period = config["default_period"]
            self.num_std = config["default_std"]  # Usually 2.0
            
            gene_config = GeneConfig(
                name=config["name"],
                min_value=config["min_period"],
                max_value=config["max_period"],
                step=config["step"],
                mutation_rate=config["mutation_rate"],
                mutation_sigma=config["mutation_range"]
            )
        else:
            self.period = gene_config.min_value
            self.num_std = 2.0  # Default value
        super().__init__(gene_config)
        
        # Override genetic operators for multiple parameters
        self._register_bollinger_operators()
        self.randomize()
        self._cache = {}
    
    def _register_bollinger_operators(self):
        """Register Bollinger-specific genetic operators"""
        gene_type_name = f"{self.__class__.__name__}Class"
        
        self.toolbox.register("attr_period", random.randint,
                            self.config.min_value,
                            self.config.max_value)
        self.toolbox.register("attr_std", random.uniform, 1.0, 3.0)
        
        self.toolbox.register("individual", tools.initCycle, 
                            getattr(creator, gene_type_name),
                            (self.toolbox.attr_period, self.toolbox.attr_std), 
                            n=1)
    
    def compute(self, data: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute Bollinger Bands for the given data using TA-Lib
        
        Args:
            data (pd.Series): Price data series
            
        Returns:
            tuple: (Upper Band, Middle Band, Lower Band)
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series")
        
        cache_key = (data.name, self.period, self.num_std)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        upper, middle, lower = talib.BBANDS(
            data,
            timeperiod=int(self.period),
            nbdevup=self.num_std,
            nbdevdn=self.num_std,
            matype=talib.MA_Type.SMA
        )
        
        result = (
            pd.Series(upper, index=data.index),
            pd.Series(middle, index=data.index),
            pd.Series(lower, index=data.index)
        )
        
        self._cache[cache_key] = result
        return result
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of Bollinger Bands parameters
        Prefers standard parameters (20 period, 2 std)
        """
        period, num_std = individual
        
        # Optimal parameters
        optimal_period = 20
        optimal_std = 2.0
        
        # Calculate distance from optimal parameters
        period_diff = abs(period - optimal_period) / optimal_period
        std_diff = abs(num_std - optimal_std) / optimal_std
        
        # Combined fitness score
        fitness = 1.0 / (1.0 + period_diff + std_diff)
        
        return (fitness,)
    
    @property
    def value(self) -> tuple[int, float]:
        """Get the current Bollinger Bands parameters"""
        return (self.period, self.num_std)
    
    @value.setter
    def value(self, new_value: list) -> None:
        """Set Bollinger Bands parameters with validation"""
        if isinstance(new_value, list) and len(new_value) == 1 and isinstance(new_value[0], list):
            new_value = new_value[0]  # Unwrap nested list from DEAP
            
        if len(new_value) != 2:
            raise ValueError("Bollinger Bands requires two parameters")
        
        period, num_std = new_value
        self.period = int(self.validate_and_clip_value(period))
        self.num_std = max(1.0, min(3.0, float(num_std)))  # Clip std between 1 and 3
        self._value = [period, num_std]  # Update internal DEAP value
    
    def mutate(self) -> None:
        """Mutate the gene using DEAP's mutation operator"""
        self._value, = self.toolbox.mutate(self._value)
        # Clip values and update period and num_std
        self.period = int(self.validate_and_clip_value(self._value[0]))
        self.num_std = max(1.0, min(3.0, float(self._value[1])))
        self._value = [self.period, self.num_std]

    def to_dict(self) -> dict:
        """Dictionary representation with Bollinger-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "period": int(self.period),
            "standard_deviations": float(self.num_std),
            "indicator_type": "Bollinger Bands",
            "library": "TA-Lib"
        })
        return base_dict
