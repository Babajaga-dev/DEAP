from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from ..utils.config_loader import ConfigLoader
from ..genes.base_gene import BaseGene

@dataclass
class StrategyConfig:
    """Configuration for strategy parameters"""
    name: str
    position_sizing: Dict[str, Any]
    indicators: List[str]
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    risk_management: Dict[str, Any]
    option_preferences: Dict[str, Any]
    risk_free_rate: float = 0.0
    transaction_costs: float = 0.0
    position_size: float = 1.0
    description: Optional[str] = None
    version: str = "1.0.0"

@dataclass
class StrategyMetadata:
    """Metadata for strategy configuration"""
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    indicators: List[str] = None
    
class BaseStrategy(ABC):
    @property
    def genes(self) -> Dict[str, BaseGene]:
        """Get strategy genes"""
        return self._genes
        
    def __init__(self, config_loader: ConfigLoader = None):
        self.fitness = None  # Per DEAP
        """
        Initialize strategy with configuration
        
        Args:
            config_loader (ConfigLoader, optional): Configuration loader instance
        """
        if config_loader is None:
            config_loader = ConfigLoader()
            
        self.config_loader = config_loader
        self._load_configuration(self.__class__.__name__.lower().replace('strategy', ''))
        
        self._initialize_genes()
        
        self._signals = pd.Series()
        self._positions = pd.Series()
        
    def _load_configuration(self, name: str) -> None:
        """Load strategy configuration from config files"""
        # Load strategy specific configuration
        strategy_name = name.replace('_', '')  # Remove underscores for matching
        config_dict = self.config_loader.get_strategy_config(strategy_name)
        risk_params = self.config_loader.get_risk_params()
        
        self.strategy_config = StrategyConfig(
            name=config_dict["name"],
            position_sizing=config_dict.get("position_sizing", {}),
            indicators=config_dict.get("indicators", []),
            entry_conditions=config_dict.get("entry_conditions", {}),
            exit_conditions=config_dict.get("exit_conditions", {}),
            risk_management=config_dict.get("risk_management", {}),
            option_preferences=config_dict.get("option_preferences", {}),
            risk_free_rate=risk_params.get("risk_free_rate", 0.0),
            transaction_costs=risk_params.get("transaction_costs", 0.0),
            position_size=config_dict.get("position_size", 1.0),
            description=config_dict.get("description"),
            version=config_dict.get("version", "1.0.0")
        )
        
        # Load general risk parameters
        self.risk_params = self.config_loader.get_risk_params()
        
        # Load backtest parameters
        self.backtest_config = self.config_loader.get_backtest_config()
        
        # Setup metadata
        self.metadata = StrategyMetadata(
            name=self.strategy_config.name,
            description=self.strategy_config.description,
            indicators=self.strategy_config.indicators
        )
    
    @abstractmethod
    def _initialize_genes(self) -> None:
        """Initialize strategy genes based on configuration"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data
        
        Args:
            data: Market data including OHLCV
            
        Returns:
            Series of signals where:
            1 = Long
            0 = Neutral
            -1 = Short
        """
        pass

    def calculate_position_sizes(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Calculate position sizes based on strategy configuration
        
        Args:
            signals: Trading signals
            data: Market data for position sizing calculations
        """
        sizing_config = self.strategy_config.position_sizing
        method = sizing_config.get("method", "fixed")
        base_size = sizing_config.get("base_size", 1.0)
        
        if method == "fixed":
            return signals * base_size
        
        elif method == "volatility_adjusted":
            vol_multiplier = sizing_config.get("volatility_multiplier", 1.0)
            volatility = data['close'].pct_change().rolling(window=20).std()
            return signals * base_size * (1 / (volatility * vol_multiplier))
        
        elif method == "kelly":
            # Implementare Kelly criterion
            pass
        
        return signals * base_size

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate strategy performance"""
        self._signals = self.generate_signals(data)
        self._positions = self.calculate_position_sizes(self._signals, data)
        
        returns = self.calculate_returns(data)
        metrics = self.calculate_metrics(returns)
        
        return metrics

    def calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns including transaction costs"""
        price_returns = data['close'].pct_change()
        strategy_returns = self._positions.shift(1) * price_returns
        
        # Apply transaction costs from configuration
        costs = self.backtest_config.get("costs", {})
        commission = costs.get("commission", {}).get("value", 0.001)
        
        trades = self._positions.diff().abs() * commission
        strategy_returns = strategy_returns - trades
        
        return strategy_returns

    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics based on configuration"""
        metrics_config = self.backtest_config.get("metrics", {})
        metrics = {}
        
        # Basic metrics
        if "total_return" in metrics_config.get("basic", []):
            metrics["total_return"] = (1 + returns).prod() - 1
            
        if "annual_return" in metrics_config.get("basic", []):
            metrics["annual_return"] = (1 + metrics["total_return"]) ** (252 / len(returns)) - 1
            
        # Always calculate volatility as it's needed for other metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        metrics["volatility"] = volatility
        
        if "sharpe_ratio" in metrics_config.get("basic", []):
            excess_returns = returns - self.risk_params["risk_free_rate"] / 252
            std = returns.std()
            if std > 0:
                sharpe = np.sqrt(252) * excess_returns.mean() / std
            else:
                sharpe = 0  # Se la deviazione standard è zero, non c'è rischio quindi Sharpe = 0
            metrics["sharpe_ratio"] = sharpe
            
        if "max_drawdown" in metrics_config.get("basic", []):
            metrics["max_drawdown"] = self.calculate_max_drawdown(returns)
        
        return metrics

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
    
    @abstractmethod
    def mutate(self) -> None:
        """Mutate strategy genes"""
        pass
    
    @abstractmethod
    def crossover(self, other: 'BaseStrategy') -> Tuple['BaseStrategy', 'BaseStrategy']:
        """Perform crossover with another strategy"""
        pass
        
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        return bool(self.strategy_config.name and 
                   self.strategy_config.position_sizing and 
                   self.strategy_config.indicators)
        
    def to_dict(self) -> Dict:
        """Convert strategy to dictionary representation"""
        return {
            'name': self.metadata.name,
            'version': self.metadata.version,
            'description': self.metadata.description,
            'indicators': self.metadata.indicators,
            'genes': {name: gene.to_dict() for name, gene in self._genes.items()},
            'config': {
                'name': self.strategy_config.name,
                'version': self.strategy_config.version,
                'description': self.strategy_config.description,
                'position_sizing': self.strategy_config.position_sizing,
                'indicators': self.strategy_config.indicators
            }
        }
