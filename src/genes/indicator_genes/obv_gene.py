from typing import Any, Tuple
import pandas as pd
import numpy as np
import talib
from deap import tools, creator, base
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class OBVGene(BaseGene):
    """
    Gene class for On Balance Volume (OBV) indicator using TA-Lib and DEAP.
    OBV accumulates volume on up days and subtracts it on down days.
    """
    
    def __init__(self, config_loader: ConfigLoader = None):
        """
        Initialize OBV gene with configuration
        
        Args:
            config_loader (ConfigLoader, optional): Configuration loader instance
        """
        if config_loader is None:
            config_loader = ConfigLoader()
            
        self._config_loader = config_loader  # Store for crossover operations
        config = config_loader.get_indicator_config("obv")
        
        # OBV doesn't have a period parameter, but we'll use a smoothing period
        # for the signal line (similar to MACD signal)
        gene_config = GeneConfig(
            name=config["name"],
            min_value=config["signal_period"]["min"],
            max_value=config["signal_period"]["max"],
            step=config["step"],
            mutation_rate=config["mutation_rate"],
            mutation_sigma=config["mutation_range"]
        )
        super().__init__(gene_config)
        # Override mutation operator for integer values
        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=max(1, (config["signal_period"]["max"] - config["signal_period"]["min"]) * 0.1),  # Ensure reasonable mutation range
            indpb=1.0  # Ensure mutation happens
        )
        self.randomize()
        self._cache = {}
    
    def compute(self, close: pd.Series, volume: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Compute OBV and its signal line for the given data using TA-Lib
        
        Args:
            close (pd.Series): Close price data series
            volume (pd.Series): Volume data series
            
        Returns:
            tuple: (OBV line, Signal line)
        """
        if not all(isinstance(x, pd.Series) for x in [close, volume]):
            raise ValueError("All inputs must be pandas Series")
        
        cache_key = (close.name, volume.name, self.value)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate raw OBV
        obv = pd.Series(
            talib.OBV(close, volume),
            index=close.index
        )
        
        # Calculate signal line (EMA of OBV)
        signal = pd.Series(
            talib.EMA(obv, timeperiod=int(self.value)),
            index=close.index
        )
        
        result = (obv, signal)
        self._cache[cache_key] = result
        return result
    
    def compute_trigger(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Compute OBV trading triggers
        
        Args:
            close (pd.Series): Close price data series
            volume (pd.Series): Volume data series
            
        Returns:
            pd.Series: Trading signals (1: buy, -1: sell, 0: neutral)
        """
        obv, signal = self.compute(close, volume)
        
        # Generate triggers when OBV crosses its signal line
        triggers = pd.Series(0, index=close.index)
        triggers[obv > signal] = 1  # Bullish
        triggers[obv < signal] = -1  # Bearish
        
        return triggers
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of an individual (signal period)
        Prefers standard periods around 20 days
        """
        signal_period = int(individual[0])
        
        # Optimal signal period
        optimal_period = 20
        
        # Calculate distance from optimal period
        distance = abs(signal_period - optimal_period)
        fitness = 1.0 / (1.0 + distance)
        
        return (fitness,)
    
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
    
    def validate_and_clip_value(self, value: float) -> float:
        """Ensure integer periods"""
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
    
    def to_dict(self) -> dict:
        """Dictionary representation with OBV-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "signal_period": int(self.value),
            "indicator_type": "OBV",
            "library": "TA-Lib",
            "description": "On Balance Volume with signal line"
        })
        return base_dict
    
    def analyze_divergence(self, close: pd.Series, volume: pd.Series, 
                         window: int = 20) -> pd.Series:
        """
        Analyze price-volume divergence
        
        Args:
            close (pd.Series): Close price data series
            volume (pd.Series): Volume data series
            window (int): Lookback window for divergence analysis
            
        Returns:
            pd.Series: Divergence signals (1: bullish, -1: bearish, 0: none)
        """
        obv, _ = self.compute(close, volume)
        
        # Calculate price and OBV changes
        price_change = close.diff(window)
        obv_change = obv.diff(window)
        
        # Initialize divergence series
        divergence = pd.Series(0, index=close.index)
        
        # Bullish divergence: price down, OBV up
        divergence[(price_change < 0) & (obv_change > 0)] = 1
        
        # Bearish divergence: price up, OBV down
        divergence[(price_change > 0) & (obv_change < 0)] = -1
        
        return divergence
        
    def crossover(self, other: 'OBVGene') -> tuple['OBVGene', 'OBVGene']:
        """
        Perform crossover with another OBV gene
        
        Args:
            other (OBVGene): Another OBV gene instance for crossover
            
        Returns:
            tuple: Two new OBV gene instances (children)
        """
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
