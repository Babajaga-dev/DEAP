from typing import Any, Tuple
import pandas as pd
import talib
from deap import tools, creator, base
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class OBVGene(BaseGene):
    """
    Gene class for On Balance Volume (OBV) indicator using TA-Lib and DEAP.
    OBV accumulates volume on up days and subtracts it on down days.
    """
    
    def __init__(self):
        """Initialize OBV gene with configuration"""
        config = ConfigLoader.get_indicator_config("obv")
        
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
        self.toolbox.register("mutate", tools.mutGaussian, 
                            mu=0, sigma=self.config.mutation_sigma, 
                            indpb=self.config.mutation_rate)
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
    
    def validate_and_clip_value(self, value: float) -> float:
        """Ensure integer periods"""
        clipped = super().validate_and_clip_value(value)
        return round(clipped)
    
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