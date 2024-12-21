from typing import Any, Tuple, Optional
import random
import pandas as pd
import numpy as np
import talib
from deap import tools
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class ATRGene(BaseGene):
    """Gene class for Average True Range indicator using TA-Lib and DEAP"""
    
    def __init__(self, config: Optional[GeneConfig] = None):
        """Initialize ATR gene with configuration"""
        if config is None:
            config_loader = ConfigLoader()
            indicator_config = config_loader.get_indicator_config("atr")
            config = GeneConfig(
                name=indicator_config["name"],
                min_value=float(indicator_config["min_period"]),
                max_value=float(indicator_config["max_period"]),
                step=float(indicator_config["step"]),
                mutation_rate=float(indicator_config["mutation_rate"]),
                mutation_sigma=float(indicator_config["mutation_range"])
            )
        super().__init__(config)
        self._cache = {}

    def _register_gene(self):
        """Override to use integer-specific operators"""
        super()._register_gene()
        # Usa float per la generazione ma assicura che siano interi
        self.toolbox.register("attr_float", lambda: float(random.randint(
            int(self.config.min_value),
            int(self.config.max_value)
        )))
        # Usa mutazione gaussiana ma arrotonda il risultato
        self.toolbox.register("mutate", tools.mutGaussian,
                            mu=0, sigma=5.0,  # Sigma più grande per permettere più variazione
                            indpb=0.5)  # Probabilità più alta di mutazione
        # Usa blend crossover ma arrotonda il risultato
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def randomize(self) -> None:
        """Initialize the gene with a random valid value"""
        super().randomize()
        # Assicura che il valore iniziale sia un float intero
        self._value = [float(round(self._value[0]))]

    def mutate(self) -> None:
        """Mutate the gene using DEAP's mutation operator"""
        mutated, = self.toolbox.mutate(self._value)
        self._value = [float(round(self.validate_and_clip_value(mutated[0])))]

    def crossover(self, other: 'BaseGene') -> tuple['BaseGene', 'BaseGene']:
        """Perform crossover with another gene"""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot crossover with gene of different type: {type(other)}")
        
        # Create new instances with the same configuration
        child1 = self.__class__(self.config)
        child2 = self.__class__(self.config)
        
        # Perform crossover using DEAP's operator
        offspring1, offspring2 = self.toolbox.mate([float(self.value)], [float(other.value)])
        
        # Set and validate the values
        child1.value = float(round(offspring1[0]))
        child2.value = float(round(offspring2[0]))
        
        return child1, child2

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
        
        # Crea una chiave di cache basata sui dati di input
        cache_key = (
            high.name,
            self.value,
            id(high),  # Usa l'id dell'oggetto per identificare univocamente la serie
            id(low),
            id(close)
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        period = int(self.value)
        
        # Assicura che i dati siano float
        high = high.astype(float)
        low = low.astype(float)
        close = close.astype(float)
        
        # Calcola l'ATR usando TA-Lib
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
        """Ensure integer periods within bounds"""
        clipped = super().validate_and_clip_value(value)
        rounded = round(clipped)
        # Ensure the rounded value stays within bounds
        return float(max(min(rounded, self.config.max_value), self.config.min_value))
    
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
