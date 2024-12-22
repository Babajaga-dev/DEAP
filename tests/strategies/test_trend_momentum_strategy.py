import pytest
import pandas as pd
import numpy as np
from src.strategies.trend_momentum_strategy import TrendMomentumStrategy
from src.strategies.base_strategy import StrategyConfig
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def config():
    return StrategyConfig(
        name="Trend Momentum Strategy",
        position_sizing={
            "method": "fixed",
            "base_size": 1.0
        },
        indicators=["rsi", "macd", "bollinger"],
        risk_free_rate=0.02,
        transaction_costs=0.001,
        position_size=1.0,
        entry_conditions={
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_threshold": 0,
            "bollinger_deviation": 2.0
        },
        exit_conditions={
            "profit_target": 0.05,
            "stop_loss": 0.02,
            "trailing_stop": 0.015
        },
        risk_management={
            "max_positions": 5,
            "max_correlation": 0.7
        },
        option_preferences={}  # Non utilizzato per TrendMomentumStrategy
    )

@pytest.fixture
def market_data():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Create trending data with volatility
    trend = np.linspace(100, 150, 100)  # Uptrend
    volatility = np.random.normal(0, 2, 100)
    close = trend + volatility
    high = close + abs(volatility)
    low = close - abs(volatility)
    volume = np.random.normal(1000000, 100000, 100)
    
    return pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

@pytest.fixture
def strategy(config, monkeypatch):
    """Create strategy with mocked config loader"""
    def mock_get_risk_params():
        return {"risk_free_rate": 0.02}
        
    def mock_get_backtest_config():
        return {
            "costs": {
                "commission": {"value": 0.001}
            },
            "metrics": {
                "basic": ["total_return", "annual_return", "sharpe_ratio", "max_drawdown"]
            }
        }

    config_loader = ConfigLoader()
    monkeypatch.setattr(config_loader, "get_risk_params", mock_get_risk_params)
    monkeypatch.setattr(config_loader, "get_backtest_config", mock_get_backtest_config)
    monkeypatch.setattr(config_loader, "get_strategy_config", lambda x: {
        "name": "Trend Momentum Strategy",
        "position_sizing": {"method": "fixed", "base_size": 1.0},
        "indicators": ["rsi", "macd", "bollinger"],
        "entry_conditions": {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_threshold": 0,
            "bollinger_deviation": 2.0
        },
        "exit_conditions": {
            "profit_target": 0.05,
            "stop_loss": 0.02,
            "trailing_stop": 0.015
        },
        "risk_management": {
            "max_positions": 5,
            "max_correlation": 0.7
        },
        "option_preferences": {}
    })

    return TrendMomentumStrategy(config_loader=config_loader)

class TestTrendMomentumStrategy:
    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy is not None
        assert len(strategy.genes) == 3
        assert 'rsi' in strategy.genes
        assert 'macd' in strategy.genes
        assert 'bollinger' in strategy.genes
    
    def test_generate_signals(self, strategy, market_data):
        """Test signal generation"""
        signals = strategy.generate_signals(market_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(market_data)
        assert set(signals.unique()).issubset({-1, 0, 1})
        
        # Test signals distribution
        signal_counts = signals.value_counts()
        assert 0 in signal_counts  # Should have some neutral positions
    
    def test_calculate_positions(self, strategy, market_data):
        """Test position calculation"""
        signals = strategy.generate_signals(market_data)
        positions = strategy.calculate_position_sizes(signals, market_data)
        
        assert isinstance(positions, pd.Series)
        assert len(positions) == len(market_data)
        assert abs(positions).max() <= strategy.strategy_config.position_size
    
    def test_evaluate(self, strategy, market_data):
        """Test strategy evaluation"""
        metrics = strategy.evaluate(market_data)
        
        assert isinstance(metrics, dict)
        required_metrics = {
            'total_return', 'annual_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown'
        }
        assert all(metric in metrics for metric in required_metrics)
        
        # Test metric values are reasonable
        assert -1 <= metrics['total_return'] <= 10
        assert -1 <= metrics['annual_return'] <= 10
        assert 0 <= metrics['volatility'] <= 1
        assert -10 <= metrics['sharpe_ratio'] <= 10
        assert -1 <= metrics['max_drawdown'] <= 0
    
    def test_calculate_returns(self, strategy, market_data):
        """Test returns calculation including transaction costs"""
        strategy._signals = strategy.generate_signals(market_data)
        strategy._positions = strategy.calculate_position_sizes(strategy._signals, market_data)
        returns = strategy.calculate_returns(market_data)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(market_data)
        # Verifica che i rendimenti siano calcolati correttamente
        assert not returns.isnull().all()  # Non dovrebbero essere tutti NaN
        assert len(returns[returns != 0]) == len(returns[strategy._positions.shift(1) != 0])  # Rendimenti non zero solo quando ci sono posizioni
        
        # Test impact of transaction costs
        trades = strategy._positions.diff().abs()
        if trades.any():  # Se ci sono trades
            assert (trades * strategy.strategy_config.transaction_costs > 0).any()
    
    def test_mutate(self, strategy):
        """Test mutation of strategy genes"""
        original_values = {
            name: gene.value 
            for name, gene in strategy.genes.items()
        }
        
        strategy.mutate()
        
        # At least one gene should change
        current_values = {
            name: gene.value 
            for name, gene in strategy.genes.items()
        }
        assert any(
            original_values[name] != current_values[name] 
            for name in original_values
        )
    
    def test_crossover(self, strategy, config):
        """Test crossover operation"""
        other = TrendMomentumStrategy(config)
        
        child1, child2 = strategy.crossover(other)
        
        assert isinstance(child1, TrendMomentumStrategy)
        assert isinstance(child2, TrendMomentumStrategy)
        assert len(child1.genes) == len(strategy.genes)
        assert len(child2.genes) == len(strategy.genes)
        
        # Test gene inheritance
        for key in strategy.genes:
            assert key in child1.genes
            assert key in child2.genes
            # I valori dei geni possono essere numeri singoli o liste di numeri (es: MACD)
            value1 = child1.genes[key].value
            value2 = child2.genes[key].value
            if isinstance(value1, (list, tuple)):
                assert all(isinstance(v, (int, float)) for v in value1)
            else:
                assert isinstance(value1, (int, float))
            if isinstance(value2, (list, tuple)):
                assert all(isinstance(v, (int, float)) for v in value2)
            else:
                assert isinstance(value2, (int, float))
    
    def test_crossover_validation(self, strategy):
        """Test crossover with invalid strategy type"""
        class DummyStrategy(TrendMomentumStrategy):
            pass
        
        # Creiamo una classe che non è una TrendMomentumStrategy
        class OtherStrategy:
            pass
        
        other = OtherStrategy()
        
        with pytest.raises(ValueError):
            strategy.crossover(other)
    
    def test_to_dict(self, strategy):
        """Test strategy serialization"""
        result = strategy.to_dict()
        
        assert isinstance(result, dict)
        assert 'name' in result
        assert 'genes' in result
        assert 'params' in result
        
        # Check genes serialization
        assert len(result['genes']) == len(strategy.genes)
        for gene_dict in result['genes'].values():
            assert 'value' in gene_dict
            assert 'type' in gene_dict
    
    def test_strategy_with_extreme_data(self, strategy):
        """Test strategy behavior with extreme market conditions"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create extreme trending data
        extreme_up = pd.DataFrame({
            'close': np.exp(np.linspace(0, 1, 50)),  # Exponential growth
            'high': np.exp(np.linspace(0, 1.1, 50)),
            'low': np.exp(np.linspace(0, 0.9, 50)),
            'volume': np.random.normal(1000000, 100000, 50)
        }, index=dates)
        
        signals_up = strategy.generate_signals(extreme_up)
        assert not signals_up.isnull().any()  # Should handle extreme trends
        
        # Test strategy with flat market
        flat_market = pd.DataFrame({
            'close': np.ones(50) * 100,
            'high': np.ones(50) * 101,
            'low': np.ones(50) * 99,
            'volume': np.random.normal(1000000, 100000, 50)
        }, index=dates)
        
        signals_flat = strategy.generate_signals(flat_market)
        assert not signals_flat.isnull().any()  # Should handle flat markets
