from typing import Dict, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from ..utils.config_loader import ConfigLoader
from ..genes.base_gene import GeneConfig
from ..genes.indicator_genes.rsi_gene import RSIGene
from ..genes.indicator_genes.bollinger_gene import BollingerGene
from ..genes.indicator_genes.atr_gene import ATRGene

class OptionsStrategy(BaseStrategy):
    def __init__(self, config_loader: ConfigLoader = None):
        """Initialize OptionsStrategy with configuration"""
        super().__init__(config_loader)

    def _initialize_genes(self) -> None:
        """Initialize strategy genes from configuration"""
        self._genes = {}
        
        # Initialize only the indicators specified in configuration
        for indicator in self.metadata.indicators:
            if indicator == "rsi":
                self._genes["rsi"] = RSIGene(self.config_loader)
            elif indicator == "bollinger":
                self._genes["bollinger"] = BollingerGene(self.config_loader)
            elif indicator == "atr":
                self._genes["atr"] = ATRGene(self.config_loader)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on configured conditions"""
        signals = pd.Series(0, index=data.index)
        
        # Load entry conditions from configuration
        entry_conditions = self.strategy_config.entry_conditions
        
        # RSI conditions
        rsi = self._genes["rsi"].compute(data["close"]).fillna(50)  # Neutral RSI for NaN
        rsi_thresholds = entry_conditions.get("rsi_thresholds", {})
        oversold = rsi < rsi_thresholds.get("oversold", 45)  # Aumentato ulteriormente soglia oversold
        overbought = rsi > rsi_thresholds.get("overbought", 55)  # Ridotto ulteriormente soglia overbought
        
        # Debug RSI
        print(f"RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
        print(f"Oversold signals: {oversold.sum()}")
        print(f"Overbought signals: {overbought.sum()}")
        
        # Volatility conditions using ATR
        atr = self._genes["atr"].compute(data["high"], data["low"], data["close"]).fillna(0)
        vol_conditions = entry_conditions.get("volatility_conditions", {})
        min_atr = vol_conditions.get("min_atr", 0.01)  # Ridotto min ATR
        
        # Calcola la media mobile dell'ATR per confronto
        atr_ma = atr.rolling(window=20, min_periods=1).mean()
        high_volatility = (atr > min_atr) & (atr > atr_ma * 0.3)  # Ridotto ulteriormente il requisito di volatilità
        
        # Debug ATR
        print(f"ATR range: {atr.min():.4f} - {atr.max():.4f}")
        print(f"High volatility signals: {high_volatility.sum()}")
        
        # Price conditions using Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._genes["bollinger"].compute(data["close"])
        # Gestisci i NaN nelle bande
        bb_upper = bb_upper.ffill()
        bb_middle = bb_middle.ffill()
        bb_lower = bb_lower.ffill()
        
        price_conditions = entry_conditions.get("price_conditions", {})
        bb_threshold = price_conditions.get("bb_threshold", 0.5)  # Percentuale della banda
        
        # Semplifica la logica delle bande
        price_above_upper = data["close"] > bb_upper
        price_below_lower = data["close"] < bb_lower
        
        # Usa il threshold per filtrare i segnali deboli
        bb_width = (bb_upper - bb_lower) / bb_middle
        significant_move = bb_width > bb_width.rolling(window=20).mean() * bb_threshold
        
        # Debug Bollinger
        bb_width = (bb_upper - bb_lower) / bb_middle
        print(f"BB width range: {bb_width.min():.2f} - {bb_width.max():.2f}")
        print(f"Price vs Upper: {((data['close'] - bb_upper) / bb_middle).max():.2f}")
        print(f"Price vs Lower: {((bb_lower - data['close']) / bb_middle).max():.2f}")
        print(f"Price above upper band signals: {price_above_upper.sum()}")
        print(f"Price below lower band signals: {price_below_lower.sum()}")
        
        # Generate signals based on RSI o price conditions con conferma delle bande
        buy_puts = (overbought | price_above_upper) & significant_move & (high_volatility | (atr > min_atr))
        buy_calls = (oversold | price_below_lower) & significant_move & (high_volatility | (atr > min_atr))
        
        signals[buy_puts] = -1  # Buy puts (bearish)
        signals[buy_calls] = 1   # Buy calls (bullish)
        
        # Apply risk management rules
        signals = self._apply_risk_management(signals, data)
        
        return signals

    def _apply_risk_management(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply risk management rules from configuration"""
        risk_config = self.strategy_config.risk_management
        
        # Maximum exposure limits
        max_positions = risk_config.get("max_positions")
        if max_positions:
            rolling_positions = signals.rolling(window=max_positions).count()
            signals[rolling_positions >= max_positions] = 0
        
        # Volatility filters - limita l'esposizione quando la volatilità è troppo alta
        max_vega_exposure = risk_config.get("max_vega_exposure")
        if max_vega_exposure:
            atr = self._genes["atr"].compute(data["high"], data["low"], data["close"])
            extreme_vol = atr > atr.rolling(20).mean() * (1 + max_vega_exposure)  # Modifica la logica
            signals[extreme_vol] = 0  # Annulla i segnali solo in caso di volatilità estrema
        
        return signals

    def calculate_position_sizes(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes based on volatility and risk parameters"""
        sizing_config = self.strategy_config.position_sizing
        base_size = sizing_config.get("base_size", 1.0)
        
        # Adjust size based on volatility
        if sizing_config.get("method") == "volatility_adjusted":
            atr = self._genes["atr"].compute(data["high"], data["low"], data["close"])
            vol_multiplier = sizing_config.get("volatility_multiplier", 1.0)
            position_sizes = signals * base_size * (1 / (atr * vol_multiplier)).fillna(1.0)
        else:
            position_sizes = signals * base_size
        
        # Gestisci i NaN nelle posizioni
        position_sizes = position_sizes.fillna(0)
            
        # Apply maximum position size limit
        max_size = self.strategy_config.risk_management.get("max_position_size", float('inf'))
        position_sizes = position_sizes.clip(-max_size, max_size)
        
        return position_sizes

    def mutate(self) -> None:
        """Mutate strategy genes with configured probability"""
        for gene in self._genes.values():
            gene.mutate()

    def crossover(self, other: 'OptionsStrategy') -> Tuple['OptionsStrategy', 'OptionsStrategy']:
        """Perform crossover with another strategy instance"""
        if not isinstance(other, OptionsStrategy):
            raise ValueError("Can only crossover with another OptionsStrategy")
        
        child1 = OptionsStrategy(self.config_loader)
        child2 = OptionsStrategy(self.config_loader)
        
        for key in self._genes:
            c1_gene, c2_gene = self._genes[key].crossover(other._genes[key])
            child1._genes[key] = c1_gene
            child2._genes[key] = c2_gene
        
        return child1, child2

    def get_option_preferences(self) -> Dict[str, float]:
        """Get option contract preferences from configuration"""
        return self.strategy_config.option_preferences

    def validate_signals(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply exit conditions and time-based filters"""
        exit_conditions = self.strategy_config.exit_conditions
        
        # Apply profit targets and stop losses
        profit_target = exit_conditions.get("profit_target")
        stop_loss = exit_conditions.get("stop_loss")
        if profit_target or stop_loss:
            position_value = (signals * data["close"]).cumsum()
            entry_price = position_value.where(signals != 0).ffill()
            
            if profit_target:
                profit_exit = (data["close"] / entry_price - 1) >= profit_target
                signals[profit_exit] = 0
                
            if stop_loss:
                loss_exit = (data["close"] / entry_price - 1) <= -stop_loss
                signals[loss_exit] = 0
        
        # Apply time decay threshold if configured
        time_decay_threshold = exit_conditions.get("time_decay_threshold")
        if time_decay_threshold:
            # In a real implementation, this would use actual option time decay
            signals = self._apply_time_decay(signals, time_decay_threshold)
            
        return signals
    
    def _apply_time_decay(self, signals: pd.Series, threshold: float) -> pd.Series:
        """
        Simulate time decay effect on positions
        In a real implementation, this would use actual option pricing
        """
        position_duration = signals.cumsum().where(signals != 0).ffill()
        time_decay = 1 - (position_duration / position_duration.max())
        
        # Close positions when time decay exceeds threshold
        signals[time_decay < threshold] = 0
        
        return signals

    def to_dict(self) -> Dict:
        """Convert strategy state to dictionary"""
        return {
            'name': 'OptionsStrategy',
            'genes': {name: gene.to_dict() for name, gene in self._genes.items()}
        }

    def from_dict(self, data: Dict) -> None:
        """Initialize strategy from dictionary state"""
        if data['name'] != 'OptionsStrategy':
            raise ValueError(f"Invalid strategy type: {data['name']}")
        
        # Initialize genes from saved state
        self._initialize_genes()
        for name, gene_data in data['genes'].items():
            if name in self._genes:
                self._genes[name].from_dict(gene_data)
