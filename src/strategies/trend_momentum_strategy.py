from typing import Dict, Tuple
import pandas as pd
from .base_strategy import BaseStrategy, StrategyMetadata, StrategyConfig
from ..utils.config_loader import ConfigLoader
from ..genes.indicator_genes.rsi_gene import RSIGene
from ..genes.indicator_genes.macd_gene import MACDGene
from ..genes.indicator_genes.bollinger_gene import BollingerGene

class TrendMomentumStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig = None, config_loader: ConfigLoader = None):
        """Initialize TrendMomentumStrategy with configuration"""
        # Setup metadata before calling super().__init__()
        self.metadata = StrategyMetadata(
            name="Trend Momentum Strategy",
            description=None,
            indicators=["rsi", "macd", "bollinger"]
        )
        super().__init__(config_loader)

    def _initialize_genes(self) -> None:
        """Initialize strategy genes from configuration"""
        self._genes = {}
        
        # Initialize only the indicators specified in configuration
        for indicator in self.metadata.indicators:
            if indicator == "rsi":
                self._genes["rsi"] = RSIGene()
            elif indicator == "macd":
                self._genes["macd"] = MACDGene()
            elif indicator == "bollinger":
                self._genes["bollinger"] = BollingerGene()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on configured conditions"""
        signals = pd.Series(0, index=data.index)
        
        # Load entry conditions from configuration
        entry_conditions = self.strategy_config.entry_conditions
        
        # RSI signals if configured
        if "rsi" in self._genes:
            rsi = self._genes["rsi"].compute(data["close"])
            rsi_oversold = rsi < entry_conditions["rsi_oversold"]
            rsi_overbought = rsi > entry_conditions["rsi_overbought"]
        else:
            rsi_oversold = rsi_overbought = pd.Series(False, index=data.index)
        
        # MACD signals if configured
        if "macd" in self._genes:
            macd, signal, hist = self._genes["macd"].compute(data["close"])
            macd_bullish = macd > signal + entry_conditions["macd_threshold"]
            macd_bearish = macd < signal - entry_conditions["macd_threshold"]
        else:
            macd_bullish = macd_bearish = pd.Series(False, index=data.index)
        
        # Bollinger signals if configured
        if "bollinger" in self._genes:
            bb_upper, bb_middle, bb_lower = self._genes["bollinger"].compute(data["close"])
            price_above_upper = data["close"] > bb_upper + entry_conditions["bollinger_deviation"]
            price_below_lower = data["close"] < bb_lower - entry_conditions["bollinger_deviation"]
        else:
            price_above_upper = price_below_lower = pd.Series(False, index=data.index)
        
        # Apply filters if configured
        valid_volume = data["volume"] > 0  # Semplifichiamo per ora
        
        # Generate signals based on combined conditions
        long_signals = (rsi_oversold & macd_bullish & price_below_lower & valid_volume)
        short_signals = (rsi_overbought & macd_bearish & price_above_upper & valid_volume)
        
        signals[long_signals] = 1
        signals[short_signals] = -1
        
        return signals

    def mutate(self) -> None:
        """Mutate strategy genes with configured probability"""
        for gene in self._genes.values():
            gene.mutate()

    def crossover(self, other: 'TrendMomentumStrategy') -> Tuple['TrendMomentumStrategy', 'TrendMomentumStrategy']:
        """Perform crossover with another strategy instance"""
        if not isinstance(other, TrendMomentumStrategy):
            raise ValueError("Can only crossover with another TrendMomentumStrategy")
        
        child1 = TrendMomentumStrategy(config_loader=self.config_loader)
        child2 = TrendMomentumStrategy(config_loader=self.config_loader)
        
        for key in self._genes:
            c1_gene, c2_gene = self._genes[key].crossover(other._genes[key])
            child1._genes[key] = c1_gene
            child2._genes[key] = c2_gene
        
        return child1, child2

    def to_dict(self) -> Dict:
        """Convert strategy to dictionary representation"""
        base_dict = super().to_dict()
        base_dict['params'] = {
            'entry_conditions': self.strategy_config.entry_conditions,
            'exit_conditions': self.strategy_config.exit_conditions,
            'risk_management': self.strategy_config.risk_management,
            'position_size': self.strategy_config.position_size,
            'risk_free_rate': self.strategy_config.risk_free_rate,
            'transaction_costs': self.strategy_config.transaction_costs
        }
        return base_dict

    def validate_signals(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply additional validation rules from configuration"""
        # Apply profit targets and stop losses if configured
        position_value = (signals * data["close"]).cumsum()
        entry_price = position_value.where(signals != 0).ffill()
        
        profit_target = self.strategy_config.exit_conditions["profit_target"]
        stop_loss = self.strategy_config.exit_conditions["stop_loss"]
        
        profit_exit = (data["close"] / entry_price - 1) >= profit_target
        loss_exit = (data["close"] / entry_price - 1) <= -stop_loss
        
        signals[profit_exit | loss_exit] = 0
        
        return signals
