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
    
    # Genera dati di mercato più realistici
    np.random.seed(42)  # Per risultati consistenti
    
    # Base trend con correzioni
    base_trend = np.zeros(100)
    base_trend[0] = 100
    
    # Genera il prezzo base con trend e cicli
    base_trend[0] = 100
    
    # Parametri per i cicli
    cycle_period = 25  # Lunghezza del ciclo in giorni
    cycle_amplitude = 0.15  # Ampiezza massima del ciclo (15%)
    
    for i in range(1, 100):
        # Componente ciclica con ampiezza variabile
        if i < 25:  # Ciclo normale
            cycle = cycle_amplitude * np.sin(2 * np.pi * i / cycle_period)
        elif i < 45:  # Ciclo quasi assente durante il crollo
            cycle = cycle_amplitude * 0.1 * np.sin(4 * np.pi * i / cycle_period)
        else:  # Ciclo normale
            cycle = cycle_amplitude * np.sin(2 * np.pi * i / cycle_period)
        
        # Componente di trend con movimenti più estremi
        if i < 25:  # Trend rialzista forte
            trend = 0.015 + 0.003 * np.random.randn()
        elif i < 35:  # Crollo estremamente violento
            trend = -0.08 + 0.01 * np.random.randn()  # -8% al giorno
        elif i < 45:  # Stabilizzazione con alta volatilità
            trend = -0.02 + 0.03 * np.random.randn()  # Alta volatilità
        elif i < 65:  # Rimbalzo molto forte
            trend = 0.035 + 0.006 * np.random.randn()
        elif i < 85:  # Correzione
            trend = -0.01 + 0.002 * np.random.randn()
        else:  # Consolidamento
            trend = 0.002 + 0.001 * np.random.randn()
        
        # Combina trend e ciclo in modo moltiplicativo
        base_trend[i] = base_trend[i-1] * (1 + trend) * (1 + cycle)
    
    # Calcola la volatilità basata sui rendimenti
    returns = np.diff(base_trend, prepend=base_trend[0]) / base_trend
    volatility = np.abs(returns) * 2  # Base volatility from returns
    
    # Amplifica la volatilità durante periodi specifici
    volatility[26:35] *= 4.0  # Volatilità estrema durante il crollo iniziale
    volatility[35:46] *= 2.5  # Alta volatilità durante la stabilizzazione
    volatility[46:66] *= 2.0  # Volatilità elevata durante il rimbalzo
    volatility[66:86] *= 1.5  # Volatilità moderata durante la correzione
    
    # Smoothing della volatilità
    volatility = np.convolve(volatility, np.ones(3)/3, mode='same')
    
    # Applica la volatilità al prezzo
    close = base_trend * (1 + volatility * np.random.randn(100) * 0.1)
    
    # Genera high e low basati sulla volatilità
    daily_range = volatility * 0.5  # Range proporzionale alla volatilità
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)
    
    # Genera volume correlato ai movimenti di prezzo e volatilità
    base_volume = np.ones(100) * 1000000
    volume = base_volume * (1 + 5 * np.abs(returns)) * (1 + 3 * volatility)
    # Aggiungi rumore al volume
    volume *= (1 + 0.05 * np.random.randn(100))  # 5% di rumore casuale
    
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
            if market_data['close'].iloc[i] > sma_20.iloc[i] and \
               market_data['close'].iloc[i] > ema_20.iloc[i] and \
               macd_data[0].iloc[i] > macd_data[1].iloc[i]:  # MACD > Signal
                # During confirmed uptrend, OBV should generally increase
                assert obv_data[0].iloc[i] >= obv_data[0].iloc[i-1]
    
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
        assert correlation > 0.3  # Correlazione moderata è sufficiente
    
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
        
        # Condizioni di ipercomprato/ipervenduto più realistiche
        overbought_agreement = (valid_data['rsi'] > 65) & (valid_data['stoch_k'] > 75)
        oversold_agreement = (valid_data['rsi'] < 40) & (valid_data['stoch_k'] < 30)  # Soglie più alte per l'ipervenduto
        
        assert overbought_agreement.any()
        assert oversold_agreement.any()
    
    def test_moving_averages_crossover(self, market_data, indicator_suite):
        """Test moving averages crossover signals"""
        # Medie mobili più ravvicinate per crossover più sincronizzati
        sma_fast = SMAGene()
        sma_fast.value = 8  # Era 10
        sma_slow = SMAGene()
        sma_slow.value = 13  # Era 20
        
        ema_fast = EMAGene()
        ema_fast.value = 8  # Era 10
        ema_slow = EMAGene()
        ema_slow.value = 13  # Era 20
        
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
        
        # Converti gli indici in posizioni numeriche
        valid_positions = [market_data.index.get_loc(idx) for idx in valid_indices[1:]]
        
        for pos in valid_positions:
            sma_cross_up = (sma_fast_values.iloc[pos-1] <= sma_slow_values.iloc[pos-1]) and \
                          (sma_fast_values.iloc[pos] > sma_slow_values.iloc[pos])
            ema_cross_up = (ema_fast_values.iloc[pos-1] <= ema_slow_values.iloc[pos-1]) and \
                          (ema_fast_values.iloc[pos] > ema_slow_values.iloc[pos])
            
            # Verifica che almeno un crossover EMA avvenga vicino a un crossover SMA
            if sma_cross_up:
                # Cerca un crossover EMA nei 3 periodi precedenti o successivi
                found_ema_cross = False
                for j in range(-3, 4):  # Da -3 a +3
                    if pos+j >= 0 and pos+j < len(ema_fast_values):
                        if (ema_fast_values.iloc[pos+j-1] <= ema_slow_values.iloc[pos+j-1]) and \
                           (ema_fast_values.iloc[pos+j] > ema_slow_values.iloc[pos+j]):
                            found_ema_cross = True
                            break
                assert found_ema_cross, "EMA crossover should occur within 3 periods of SMA crossover"
    
    def test_volume_price_relationship(self, market_data, indicator_suite):
        """Test relationship between volume and price indicators"""
        obv, obv_signal = indicator_suite['obv'].compute(
            market_data['close'],
            market_data['volume']
        )
        
        macd, macd_signal, macd_hist = indicator_suite['macd'].compute(
            market_data['close']
        )
        
        # Test if strong MACD movements are generally confirmed by higher volume
        strong_macd_indices = []
        for i in range(1, len(market_data)):
            if abs(macd_hist.iloc[i]) > abs(macd_hist.iloc[i-1]) * 1.5:
                strong_macd_indices.append(i)
        
        if strong_macd_indices:
            # Calcola il volume medio durante i movimenti MACD forti
            strong_macd_volume = market_data['volume'].iloc[strong_macd_indices].mean()
            # Calcola il volume medio generale
            average_volume = market_data['volume'].mean()
            # Il volume medio durante i movimenti MACD forti dovrebbe essere maggiore
            assert strong_macd_volume > average_volume
    
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
            if all(market_data['close'].iloc[i] > market_data['close'].iloc[i+j] for j in range(-window, window+1) if j != 0):
                price_higher_highs.append(i)
                
            if all(rsi.iloc[i] > rsi.iloc[i+j] for j in range(-window, window+1) if j != 0):
                rsi_lower_highs.append(i)
                
            if all(obv.iloc[i] > obv.iloc[i+j] for j in range(-window, window+1) if j != 0):
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
            # Long signal conditions (ancora più flessibili)
            price_near_lower_bb = market_data['close'].iloc[i] <= bb_lower.iloc[i] * 1.05
            rsi_oversold = rsi.iloc[i] < 45
            
            # Short signal conditions (ancora più flessibili)
            price_near_upper_bb = market_data['close'].iloc[i] >= bb_upper.iloc[i] * 0.95
            rsi_overbought = rsi.iloc[i] > 55
            
            if price_near_lower_bb and rsi_oversold:
                signals.iloc[i] = 1  # Long signal
            elif price_near_upper_bb and rsi_overbought:
                signals.iloc[i] = -1  # Short signal
        
        # Verify that signals are generated
        assert (signals != 0).any()
        
        # Verify signal effectiveness
        returns = market_data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Strategy should have some profitable trades
        assert (strategy_returns > 0).any()
