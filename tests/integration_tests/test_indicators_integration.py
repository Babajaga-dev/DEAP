import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.sma_gene import SMAGene
from src.genes.indicator_genes.rsi_gene import RSIGene
from src.genes.indicator_genes.macd_gene import MACDGene
from src.genes.indicator_genes.bollinger_gene import BollingerGene
from src.genes.indicator_genes.stochastic_gene import StochasticGene
from src.genes.indicator_genes.atr_gene import ATRGene
from src.genes.indicator_genes.ema_gene import EMAGene
from src.genes.indicator_genes.obv_gene import OBVGene

@pytest.fixture
def market_data():
    """Fixture that provides comprehensive market data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Create base trend
    base_trend = np.linspace(100, 150, 100)  # Overall uptrend
    
    # Add cyclical component
    cycles = 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    
    # Add volatility clusters
    volatility = np.random.normal(0, 1, 100)
    volatility[30:40] *= 2  # Higher volatility period
    volatility[60:70] *= 0.5  # Lower volatility period
    
    # Combine components
    close = base_trend + cycles + volatility
    high = close + np.abs(volatility) * 0.5
    low = close - np.abs(volatility) * 0.5
    volume = np.random.normal(1000000, 100000, 100)
    volume[close > np.roll(close, 1)] *= 1.2  # Higher volume on up days
    
    return pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

@pytest.fixture
def indicator_suite():
    """Fixture that provides all initialized indicators"""
    return {
        'sma': SMAGene(),
        'ema': EMAGene(),
        'rsi': RSIGene(),
        'macd': MACDGene(),
        'bollinger': BollingerGene(),
        'stochastic': StochasticGene(),
        'atr': ATRGene(),
        'obv': OBVGene()
    }

class TestIndicatorIntegration:
    def test_trend_confirmation(self, market_data, indicator_suite):
        """Test trend confirmation across multiple indicators"""
        # Calculate indicator values
        sma_20 = indicator_suite['sma'].compute(market_data['close'])
        ema_20 = indicator_suite['ema'].compute(market_data['close'])
        macd_data = indicator_suite['macd'].compute(market_data['close'])
        obv_data = indicator_suite['obv'].compute(market_data['close'], 
                                                market_data['volume'])
        
        # Check trend agreement
        for i in range(30, len(market_data)):
            if market_data['close'][i] > sma_20[i] and \
               market_data['close'][i] > ema_20[i] and \
               macd_data[0][i] > macd_data[1][i]:  # MACD > Signal
                # During confirmed uptrend, OBV should generally increase
                assert obv_data[0][i] >= obv_data[0][i-1]
    
    def test_volatility_correlation(self, market_data, indicator_suite):
        """Test correlation between volatility indicators"""
        # Calculate Bollinger Bands width and ATR
        bb_upper, bb_middle, bb_lower = indicator_suite['bollinger'].compute(
            market_data['close']
        )
        bb_width = bb_upper - bb_lower
        
        atr = indicator_suite['atr'].compute(
            market_data['high'],
            market_data['low'],
            market_data['close']
        )
        
        # Calculate correlation between BB width and ATR
        valid_data = pd.DataFrame({
            'bb_width': bb_width,
            'atr': atr
        }).dropna()
        
        correlation = valid_data['bb_width'].corr(valid_data['atr'])
        assert correlation > 0.5  # Should be positively correlated
    
    def test_momentum_confirmation(self, market_data, indicator_suite):
        """Test agreement between momentum indicators"""
        rsi = indicator_suite['rsi'].compute(market_data['close'])
        stoch_k, stoch_d = indicator_suite['stochastic'].compute(
            market_data['close'],
            market_data['high'],
            market_data['low']
        )
        
        # Check for overbought/oversold agreement
        valid_data = pd.DataFrame({
            'rsi': rsi,
            'stoch_k': stoch_k
        }).dropna()
        
        # Both indicators should agree on extreme conditions
        overbought_agreement = (valid_data['rsi'] > 70) & (valid_data['stoch_k'] > 80)
        oversold_agreement = (valid_data['rsi'] < 30) & (valid_data['stoch_k'] < 20)
        
        assert overbought_agreement.any()
        assert oversold_agreement.any()
    
    def test_moving_averages_crossover(self, market_data, indicator_suite):
        """Test moving averages crossover signals"""
        sma_fast = SMAGene()
        sma_fast.value = 10
        sma_slow = SMAGene()
        sma_slow.value = 20
        
        ema_fast = EMAGene()
        ema_fast.value = 10
        ema_slow = EMAGene()
        ema_slow.value = 20
        
        # Calculate MAs
        sma_fast_values = sma_fast.compute(market_data['close'])
        sma_slow_values = sma_slow.compute(market_data['close'])
        ema_fast_values = ema_fast.compute(market_data['close'])
        ema_slow_values = ema_slow.compute(market_data['close'])
        
        # Test crossover agreement
        valid_indices = pd.DataFrame({
            'sma_cross': sma_fast_values - sma_slow_values,
            'ema_cross': ema_fast_values - ema_slow_values
        }).dropna().index
        
        for i in valid_indices[1:]:
            sma_cross_up = (sma_fast_values[i-1] <= sma_slow_values[i-1]) and \
                          (sma_fast_values[i] > sma_slow_values[i])
            ema_cross_up = (ema_fast_values[i-1] <= ema_slow_values[i-1]) and \
                          (ema_fast_values[i] > ema_slow_values[i])
            
            # Crossovers should generally occur close to each other
            if sma_cross_up:
                assert abs(ema_fast_values[i] - ema_slow_values[i]) < \
                       abs(ema_fast_values[i-5] - ema_slow_values[i-5])
    
    def test_volume_price_relationship(self, market_data, indicator_suite):
        """Test relationship between volume and price indicators"""
        obv, obv_signal = indicator_suite['obv'].compute(
            market_data['close'],
            market_data['volume']
        )
        
        macd, macd_signal, macd_hist = indicator_suite['macd'].compute(
            market_data['close']
        )
        
        # Test if strong MACD movements are confirmed by volume
        for i in range(1, len(market_data)):
            if abs(macd_hist[i]) > abs(macd_hist[i-1]) * 1.5:  # Strong MACD movement
                # Volume should increase
                assert market_data['volume'][i] > market_data['volume'][i-1]
    
    def test_divergence_detection(self, market_data, indicator_suite):
        """Test divergence detection across indicators"""
        rsi = indicator_suite['rsi'].compute(market_data['close'])
        obv, _ = indicator_suite['obv'].compute(
            market_data['close'],
            market_data['volume']
        )
        
        price_higher_highs = []
        rsi_lower_highs = []
        obv_lower_highs = []
        
        # Detect divergences
        window = 5
        for i in range(window, len(market_data)-window):
            if all(market_data['close'][i] > market_data['close'][i+j] for j in range(-window, window+1) if j != 0):
                price_higher_highs.append(i)
                
            if all(rsi[i] > rsi[i+j] for j in range(-window, window+1) if j != 0):
                rsi_lower_highs.append(i)
                
            if all(obv[i] > obv[i+j] for j in range(-window, window+1) if j != 0):
                obv_lower_highs.append(i)
        
        # Check if divergences are detected by multiple indicators
        divergences = []
        for p_idx in price_higher_highs:
            if p_idx in rsi_lower_highs and p_idx in obv_lower_highs:
                divergences.append(p_idx)
        
        assert len(divergences) > 0  # Should detect some divergences
    
    def test_indicator_combination_strategy(self, market_data, indicator_suite):
        """Test a combined trading strategy using multiple indicators"""
        # Calculate all necessary indicators
        sma = indicator_suite['sma'].compute(market_data['close'])
        rsi = indicator_suite['rsi'].compute(market_data['close'])
        bb_upper, bb_middle, bb_lower = indicator_suite['bollinger'].compute(
            market_data['close']
        )
        
        signals = pd.Series(0, index=market_data.index)
        
        for i in range(1, len(market_data)):
            # Long signal conditions
            price_above_sma = market_data['close'][i] > sma[i]
            rsi_bullish = 30 < rsi[i] < 70
            price_near_lower_bb = market_data['close'][i] <= bb_lower[i] * 1.01
            
            # Short signal conditions
            price_below_sma = market_data['close'][i] < sma[i]
            rsi_bearish = rsi[i] > 70
            price_near_upper_bb = market_data['close'][i] >= bb_upper[i] * 0.99
            
            if price_above_sma and rsi_bullish and price_near_lower_bb:
                signals[i] = 1  # Long signal
            elif price_below_sma and rsi_bearish and price_near_upper_bb:
                signals[i] = -1  # Short signal
        
        # Verify that signals are generated
        assert (signals != 0).any()
        
        # Verify signal effectiveness
        returns = market_data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Strategy should have some profitable trades
        assert (strategy_returns > 0).any()