import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.strategies.options_strategy import OptionsStrategy
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def mock_indicators_data():
    """Create mock indicators configuration data for testing"""
    return {
        "indicators": {
            "rsi": {
                "name": "Relative Strength Index",
                "min_period": 2,
                "max_period": 50,
                "default_period": 14,
                "mutation_rate": 0.1,
                "mutation_range": 0.2,
                "step": 1
            },
            "bollinger": {
                "name": "Bollinger Bands",
                "min_period": 5,
                "max_period": 50,
                "default_period": 20,
                "default_std": 2.0,
                "mutation_rate": 0.1,
                "mutation_range": 0.2,
                "step": 1
            },
            "atr": {
                "name": "Average True Range",
                "min_period": 5,
                "max_period": 50,
                "default_period": 14,
                "mutation_rate": 0.1,
                "mutation_range": 0.2,
                "step": 1
            }
        }
    }

@pytest.fixture
def mock_backtest_data():
    """Create mock backtest configuration data for testing"""
    return {
        "backtest": {
            "general": {
                "initial_capital": 100000,
                "currency": "USD",
                "leverage_allowed": True,
                "max_leverage": 2.0,
                "margin_requirement": 0.5
            },
            "time": {
                "timeframe": "1h",
                "warmup_period": 100,
                "market_hours": {
                    "start": "09:30",
                    "end": "16:00"
                },
                "time_zone": "UTC"
            },
            "execution": {
                "slippage_model": "fixed",
                "price_impact": 0.0001,
                "latency": 100,
                "fill_probability": 0.98
            },
            "costs": {
                "commission": {
                    "type": "percentage",
                    "value": 0.001
                },
                "option_commission": 0.65
            }
        }
    }

@pytest.fixture
def mock_config_data():
    """Create mock configuration data for testing"""
    return {
        "strategies": {
            "options": {
                "name": "Options Volatility Strategy",
                "indicators": ["rsi", "bollinger", "atr"],
                "position_sizing": {
                    "method": "volatility_adjusted",
                    "base_size": 1.0,
                    "volatility_multiplier": 0.5
                },
                "entry_conditions": {
                    "rsi_thresholds": {
                        "oversold": 30,
                        "overbought": 70
                    },
                    "volatility_conditions": {
                        "min_atr": 0.02,
                        "atr_percentile": 75
                    },
                    "price_conditions": {
                        "bb_threshold": 2.0
                    }
                },
                "exit_conditions": {
                    "profit_target": 0.1,
                    "stop_loss": 0.05,
                    "time_decay_threshold": 0.7
                },
                "risk_management": {
                    "max_positions": 5,
                    "max_vega_exposure": 0.2,
                    "max_gamma_exposure": 0.1,
                    "max_position_size": 2.0
                },
                "option_preferences": {
                    "min_days_to_expiry": 30,
                    "max_days_to_expiry": 60,
                    "preferred_delta": 0.3,
                    "iv_rank_min": 0.5
                }
            },
            "general": {
                "risk_free_rate": 0.02,
                "transaction_costs": 0.001
            }
        }
    }

@pytest.fixture
def mock_market_data():
    """Create mock market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    # Crea una serie di prezzi con movimenti estremi
    closes = np.zeros(100)
    closes[0] = 100.0
    
    # Fase 1: Strong uptrend (giorni 1-30)
    for i in range(1, 30):
        if i % 5 < 4:  # 4 giorni up, 1 giorno down
            closes[i] = closes[i-1] * 1.03  # +3% up days
        else:
            closes[i] = closes[i-1] * 0.99  # -1% down day
    
    # Fase 2: Strong downtrend (giorni 31-60)
    for i in range(30, 60):
        if i % 5 < 4:  # 4 giorni down, 1 giorno up
            closes[i] = closes[i-1] * 0.97  # -3% down days
        else:
            closes[i] = closes[i-1] * 1.01  # +1% up day
    
    # Fase 3: Consolidation with high volatility (giorni 61-100)
    for i in range(60, 100):
        if i % 2 == 0:
            closes[i] = closes[i-1] * 1.02  # +2% up day
        else:
            closes[i] = closes[i-1] * 0.98  # -2% down day
    
    # Aggiungi volatilità nei punti di svolta
    volatility = np.zeros(100)
    volatility[25:35] = closes[25:35] * 0.05   # Alta volatilità nel primo punto di svolta
    volatility[55:65] = closes[55:65] * 0.05   # Alta volatilità nel secondo punto di svolta
    # Crea il DataFrame con high/low basati sulla volatilità
    data = pd.DataFrame({
        'close': closes,
        'high': closes + volatility + (closes * 0.01),  # Base volatility + 1%
        'low': closes - volatility - (closes * 0.01),   # Base volatility - 1%
        'open': closes * 0.995,  # Leggero gap all'apertura
        'volume': np.random.uniform(90000000, 120000000, 100)
    }, index=dates)
    
    # Aggiungi dati di warmup prima del periodo di test
    warmup_data = pd.DataFrame({
        'close': [closes[0]] * 20,  # Replica il primo prezzo per il warmup
        'high': [closes[0] * 1.01] * 20,
        'low': [closes[0] * 0.99] * 20,
        'open': [closes[0] * 0.995] * 20,
        'volume': np.random.uniform(90000000, 120000000, 20)
    }, index=pd.date_range(start='2023-12-12', end='2023-12-31', freq='D'))
    
    # Concatena i dati di warmup con i dati di test
    data = pd.concat([warmup_data, data])
    return data

@pytest.fixture
def config_loader(tmp_path, mock_config_data, mock_backtest_data, mock_indicators_data):
    """Create ConfigLoader with mock data"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    import yaml
    # Create strategies.yaml
    with open(config_dir / "strategies.yaml", 'w') as f:
        yaml.dump(mock_config_data, f)
    
    # Create backtest.yaml
    with open(config_dir / "backtest.yaml", 'w') as f:
        yaml.dump(mock_backtest_data, f)
        
    # Create indicators.yaml
    with open(config_dir / "indicators.yaml", 'w') as f:
        yaml.dump(mock_indicators_data, f)
    
    loader = ConfigLoader()
    loader.config_dir = str(config_dir)
    return loader

@pytest.fixture
def strategy(config_loader):
    """Create OptionsStrategy instance"""
    return OptionsStrategy(config_loader)

class TestOptionsStrategy:
    def test_initialization(self, strategy):
        """Test strategy initialization and configuration loading"""
        assert strategy is not None
        assert strategy.metadata.name == "Options Volatility Strategy"
        assert len(strategy.genes) == 3
        assert all(name in strategy.genes for name in ['rsi', 'bollinger', 'atr'])
        
    def test_generate_signals(self, strategy, mock_market_data):
        """Test signal generation logic and risk management"""
        signals = strategy.generate_signals(mock_market_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(mock_market_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

        # Calcola gli indicatori per verificare la coerenza dei segnali
        rsi = strategy._genes["rsi"].compute(mock_market_data["close"])
        bb_upper, _, bb_lower = strategy._genes["bollinger"].compute(mock_market_data["close"])
        atr = strategy._genes["atr"].compute(mock_market_data["high"], mock_market_data["low"], mock_market_data["close"])
        
        # Verifica la coerenza dei segnali con gli indicatori
        for i in range(len(signals)):
            if signals.iloc[i] == 1:  # Segnale rialzista
                # Verifica che almeno una delle condizioni di acquisto sia vera
                price_condition = mock_market_data["close"].iloc[i] < bb_lower.iloc[i]
                rsi_condition = rsi.iloc[i] < 45  # Soglia oversold
                assert price_condition or rsi_condition, f"Segnale long non valido all'indice {i}"
            
            elif signals.iloc[i] == -1:  # Segnale ribassista
                # Verifica che almeno una delle condizioni di vendita sia vera
                price_condition = mock_market_data["close"].iloc[i] > bb_upper.iloc[i]
                rsi_condition = rsi.iloc[i] > 55  # Soglia overbought
                assert price_condition or rsi_condition, f"Segnale short non valido all'indice {i}"

        # Verifica la gestione del rischio
        # Non dovrebbero esserci segnali con volatilità estrema
        extreme_vol = atr > atr.rolling(20).mean() * 2
        assert not any(signals[extreme_vol] != 0), "Segnali generati con volatilità estrema"

        # Verifica la gestione dell'esposizione complessiva
        max_positions = strategy.strategy_config.risk_management.get("max_positions", 5)
        
        # Calcola l'esposizione netta (long - short) su una finestra mobile
        net_exposure = signals.rolling(window=20, min_periods=1).sum()
        
        # L'esposizione netta non dovrebbe superare il limite di posizioni in nessuna direzione
        assert all(abs(net_exposure) <= max_positions), f"Esposizione netta eccessiva (limite: {max_positions})"
        
        # Verifica che i segnali siano distribuiti nel tempo
        signal_gaps = pd.Series(
            [len(signals[i:i+20][signals[i:i+20] != 0]) 
             for i in range(0, len(signals)-20)],
            index=signals.index[:-20]
        )
        
        # Non dovrebbero esserci troppi segnali concentrati in brevi periodi
        assert all(signal_gaps <= max_positions * 2), "Segnali troppo concentrati nel tempo"
        
    def test_position_sizing(self, strategy, mock_market_data):
        """Test position sizing calculations"""
        signals = strategy.generate_signals(mock_market_data)
        positions = strategy.calculate_position_sizes(signals, mock_market_data)
        
        assert isinstance(positions, pd.Series)
        assert len(positions) == len(signals)
        
        # Check position size limits
        max_size = strategy.strategy_config.risk_management.get("max_position_size", float('inf'))
        assert all(abs(positions) <= max_size)
        
    def test_risk_management(self, strategy, mock_market_data):
        """Test risk management rules"""
        signals = pd.Series(1, index=mock_market_data.index)  # All long signals
        signals = strategy._apply_risk_management(signals, mock_market_data)
        
        max_positions = strategy.strategy_config.risk_management.get("max_positions", 5)
        rolling_positions = signals.rolling(window=max_positions, min_periods=1).sum().fillna(0)
        assert all(rolling_positions[20:] <= max_positions)  # Check after warmup period
        
    def test_validate_signals(self, strategy, mock_market_data):
        """Test signal validation and exit conditions"""
        signals = strategy.generate_signals(mock_market_data)
        validated_signals = strategy.validate_signals(signals, mock_market_data)
        
        assert isinstance(validated_signals, pd.Series)
        assert len(validated_signals) == len(signals)
        
        # Test profit target and stop loss
        profit_target = strategy.strategy_config.exit_conditions.get("profit_target", 0.1)
        stop_loss = strategy.strategy_config.exit_conditions.get("stop_loss", 0.05)
        
        returns = (mock_market_data['close'].pct_change() * signals.shift(1)).fillna(0)
        cum_returns = returns.cumsum()
        assert all(cum_returns[20:] <= profit_target)  # No position exceeds profit target after warmup
        assert all(cum_returns[20:] >= -stop_loss)    # No position exceeds stop loss after warmup
        
    def test_option_preferences(self, strategy):
        """Test option contract preferences"""
        preferences = strategy.get_option_preferences()
        
        assert isinstance(preferences, dict)
        assert "min_days_to_expiry" in preferences
        assert "max_days_to_expiry" in preferences
        assert "preferred_delta" in preferences
        
    def test_time_decay(self, strategy):
        """Test time decay simulation"""
        signals = pd.Series([1] * 10)  # Constant position
        threshold = strategy.strategy_config.exit_conditions.get("time_decay_threshold", 0.7)
        
        decayed_signals = strategy._apply_time_decay(signals, threshold)
        assert len(decayed_signals[decayed_signals == 0]) > 0  # Some positions should be closed
        
    def test_mutate(self, strategy):
        """Test mutation of strategy genes"""
        original_values = {
            name: gene.value for name, gene in strategy.genes.items()
        }
        
        strategy.mutate()
        
        current_values = {
            name: gene.value for name, gene in strategy.genes.items()
        }
        assert any(
            original_values[name] != current_values[name] 
            for name in original_values
        )
        
    def test_crossover(self, strategy, config_loader):
        """Test crossover operation"""
        other = OptionsStrategy(config_loader)
        
        child1, child2 = strategy.crossover(other)
        
        assert isinstance(child1, OptionsStrategy)
        assert isinstance(child2, OptionsStrategy)
        assert len(child1.genes) == len(strategy.genes)
        assert len(child2.genes) == len(strategy.genes)
        
    def test_invalid_crossover(self, strategy):
        """Test crossover with invalid strategy type"""
        class DummyStrategy:
            pass
            
        with pytest.raises(ValueError):
            strategy.crossover(DummyStrategy())
            
    def test_extreme_market_conditions(self, strategy):
        """Test strategy behavior with extreme market conditions"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create highly volatile data
        volatile_data = pd.DataFrame({
            'close': np.exp(np.linspace(0, 1, 50)),  # Exponential growth
            'high': np.exp(np.linspace(0.1, 1.2, 50)),
            'low': np.exp(np.linspace(-0.1, 0.8, 50)),
            'volume': np.random.normal(1000000, 100000, 50)
        }, index=dates)
        
        signals = strategy.generate_signals(volatile_data)
        assert not signals.isnull().any()  # Should handle extreme data
        
        # Test with low volatility
        flat_data = pd.DataFrame({
            'close': np.ones(50) * 100,
            'high': np.ones(50) * 101,
            'low': np.ones(50) * 99,
            'volume': np.ones(50) * 1000000
        }, index=dates)
        
        signals_flat = strategy.generate_signals(flat_data)
        assert not signals_flat.isnull().any()  # Should handle flat markets
        assert (signals_flat == 0).all()  # Should not trade in flat markets
