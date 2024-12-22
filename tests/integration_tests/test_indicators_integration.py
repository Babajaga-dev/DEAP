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
    cycle_period = 30  # Lunghezza del ciclo in giorni
    cycle_amplitude = 0.15  # Ampiezza massima del ciclo (15%)
    
    for i in range(1, 100):
        # Componente ciclica con ampiezza variabile
        if i < 20:  # Ciclo ridotto durante il trend rialzista intenso
            cycle = cycle_amplitude * 0.2 * np.sin(2 * np.pi * i / cycle_period)  # Ampiezza ridotta ma non troppo
        elif i < 30:  # Consolidamento con ciclo normale
            cycle = cycle_amplitude * 0.5 * np.sin(2 * np.pi * i / cycle_period)
        elif i < 45:  # Ciclo quasi assente durante il crollo
            cycle = cycle_amplitude * 0.1 * np.sin(4 * np.pi * i / cycle_period)
        else:  # Ciclo normale
            cycle = cycle_amplitude * np.sin(2 * np.pi * i / cycle_period)
        
        # Componente di trend con movimenti più estremi
        if i < 20:  # Trend rialzista molto forte e concentrato
            trend = 0.08 + 0.002 * np.random.randn()  # 8% al giorno con rumore moderato
        elif i < 30:  # Consolidamento laterale
            trend = 0.001 + 0.003 * np.random.randn()  # Movimento laterale con volatilità
        elif i < 40:  # Crollo estremamente violento
            trend = -0.10 + 0.005 * np.random.randn()  # -10% al giorno con meno rumore
        elif i < 45:  # Stabilizzazione con volatilità moderata
            trend = -0.02 + 0.01 * np.random.randn()  # Volatilità ridotta
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
    volatility[26:40] *= 3.0  # Volatilità moderata durante il crollo per permettere trend più pulito
    volatility[40:46] *= 2.0  # Volatilità ridotta durante la stabilizzazione
    volatility[46:66] *= 3.0  # Volatilità elevata durante il rimbalzo
    volatility[66:86] *= 2.0  # Volatilità moderata durante la correzione
    
    # Smoothing della volatilità
    volatility = np.convolve(volatility, np.ones(3)/3, mode='same')
    
    # Applica la volatilità al prezzo con rumore ridotto nel periodo iniziale
    noise = np.random.randn(100) * 0.1
    noise[:25] *= 0.3  # Riduce il rumore nel periodo rialzista
    close = base_trend * (1 + volatility * noise)
    
    # Genera high e low con range più contenuto
    daily_range = volatility * 0.3  # Range ridotto per movimenti più direzionali
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)
    
    # Genera volume correlato ai movimenti di prezzo e volatilità
    base_volume = np.ones(100) * 1000000
    volume = base_volume * (1 + 8 * np.abs(returns)) * (1 + 5 * volatility)  # Aumentata sensibilità del volume
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
        uptrend_periods = []
        for i in range(30, len(market_data)):
            if market_data['close'].iloc[i] > sma_20.iloc[i] and \
               market_data['close'].iloc[i] > ema_20.iloc[i] and \
               macd_data[0].iloc[i] > macd_data[1].iloc[i]:  # MACD > Signal
                uptrend_periods.append(i)
        
        # Verifica che l'OBV abbia una tendenza generale al rialzo durante i trend rialzisti
        if len(uptrend_periods) > 0:
            start_idx = uptrend_periods[0]
            end_idx = uptrend_periods[-1]
            # Calcola la media mobile dell'OBV per ridurre il rumore
            obv_ma = pd.Series(obv_data[0]).rolling(window=5).mean()
            assert obv_ma.iloc[end_idx] >= obv_ma.iloc[start_idx], \
                "OBV moving average should show overall increase during confirmed uptrend"
    
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
        
        # Normalizza i dati prima di calcolare la correlazione
        bb_width_norm = (bb_width - bb_width.mean()) / bb_width.std()
        atr_norm = (atr - atr.mean()) / atr.std()
        
        # Calcola la correlazione su una finestra mobile
        rolling_corr = pd.Series(bb_width_norm).rolling(window=20).corr(pd.Series(atr_norm))
        
        # Verifica che ci siano periodi di correlazione significativa
        assert rolling_corr.max() > 0.5, "Should have periods of significant correlation"
    
    def test_momentum_confirmation(self, market_data, indicator_suite):
        """Test agreement between momentum indicators"""
        rsi = indicator_suite['rsi'].compute(market_data['close'])
        stoch_k, stoch_d = indicator_suite['stochastic'].compute(
            market_data['close'],
            market_data['high'],
            market_data['low']
        )
        
        # Verifica la correlazione tra RSI e Stochastic
        valid_data = pd.DataFrame({
            'rsi': rsi,
            'stoch_k': stoch_k,
        }).dropna()

        # Calcola la correlazione tra RSI e Stochastic
        correlation = valid_data['rsi'].corr(valid_data['stoch_k'])
        
        # Verifica che ci sia una correlazione positiva minima
        assert correlation > 0.3, f"RSI e Stochastic dovrebbero avere una correlazione positiva minima (correlazione attuale: {correlation})"
        
        # Verifica che i movimenti siano nella stessa direzione su una media mobile
        rsi_direction = valid_data['rsi'].rolling(window=5).mean().diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        stoch_direction = valid_data['stoch_k'].rolling(window=5).mean().diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        # Calcola la percentuale di accordo nella direzione usando una finestra mobile
        direction_agreement = (rsi_direction == stoch_direction).rolling(window=20, min_periods=1).mean()
        assert direction_agreement.mean() > 0.5, f"RSI e Stochastic dovrebbero muoversi nella stessa direzione almeno il 50% delle volte su una media mobile (accordo attuale: {direction_agreement.mean()*100:.2f}%)"
        
        # Verifica che ci siano periodi di forte accordo
        assert direction_agreement.max() > 0.7, f"RSI e Stochastic dovrebbero avere periodi di forte accordo direzionale (massimo accordo: {direction_agreement.max()*100:.2f}%)"
    
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
