import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.atr_gene import ATRGene

@pytest.fixture
def market_data():
    """Fixture that provides sample market data including high, low, close prices"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Crea un trend di base
    base = np.linspace(100, 110, len(dates))
    # Aggiungi volatilità casuale
    noise = np.random.normal(0, 1, len(dates))
    
    close = pd.Series(base + noise, index=dates, name='close')
    # High è sempre maggiore del close
    high = pd.Series(close + np.abs(np.random.normal(0, 1, len(dates))), 
                    index=dates, name='high')
    # Low è sempre minore del close
    low = pd.Series(close - np.abs(np.random.normal(0, 1, len(dates))), 
                   index=dates, name='low')
    
    return high, low, close

@pytest.fixture
def atr_gene():
    """Fixture that provides an ATR gene instance"""
    return ATRGene()

class TestATRGene:
    def test_initialization(self):
        """Test ATR gene initialization"""
        gene = ATRGene()
        assert isinstance(gene, ATRGene)
        assert gene.value is not None
        assert isinstance(gene.value, float)
        assert gene.value.is_integer()
    
    def test_compute(self, atr_gene, market_data):
        """Test ATR computation"""
        high, low, close = market_data
        atr_gene.value = 14  # Standard ATR period
        
        result = atr_gene.compute(high, low, close)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)
        
        # First values should be NaN
        assert pd.isna(result[:13]).all()
        
        # ATR should be positive
        valid_values = result.dropna()
        assert (valid_values > 0).all()
        
        # Test response to volatility
        high_vol_period = high.copy() * 1.1  # Increase high by 10%
        low_vol_period = low.copy() * 0.9   # Decrease low by 10%
        high_vol_result = atr_gene.compute(high_vol_period, low_vol_period, close)
        
        # ATR should be higher with increased volatility
        # Confronta le medie per essere più robusti
        assert high_vol_result.mean() > result.mean()
    
    def test_period_validation(self, atr_gene):
        """Test validation of ATR period"""
        config = atr_gene.config
        
        # Test minimum period
        atr_gene.value = config.min_value - 1
        assert atr_gene.value == config.min_value
        
        # Test maximum period
        atr_gene.value = config.max_value + 1
        assert atr_gene.value == config.max_value
    
    def test_crossover(self, atr_gene):
        """Test crossover operation"""
        parent2 = ATRGene()
        
        atr_gene.value = 14
        parent2.value = 20
        
        child1, child2 = atr_gene.crossover(parent2)
        
        for child in [child1, child2]:
            assert isinstance(child, ATRGene)
            assert isinstance(child.value, float)
            assert child.value.is_integer()
            assert child.config.min_value <= child.value <= child.config.max_value
    
    def test_mutation(self, atr_gene):
        """Test mutation operation"""
        original_value = atr_gene.value
        
        mutated = False
        for _ in range(10):
            atr_gene.mutate()
            if atr_gene.value != original_value:
                mutated = True
                break
        
        assert mutated, "Value should change after multiple mutations"
        assert isinstance(atr_gene.value, float)
        assert atr_gene.value.is_integer()
    
    def test_cache_functionality(self, atr_gene, market_data):
        """Test that caching works correctly"""
        high, low, close = market_data
        atr_gene.value = 14
        
        # First computation
        result1 = atr_gene.compute(high, low, close)
        
        # Second computation with same parameters
        result2 = atr_gene.compute(high, low, close)
        
        # Should get same object from cache
        assert result1 is result2
    
    def test_to_dict(self, atr_gene):
        """Test dictionary representation"""
        atr_gene.value = 14
        result = atr_gene.to_dict()
        
        assert result["period"] == 14
        assert result["indicator_type"] == "ATR"
        assert result["library"] == "TA-Lib"
        assert "description" in result
    
    def test_evaluate_individual(self, atr_gene):
        """Test the fitness evaluation with emphasis on standard period of 14"""
        # Test optimal period
        optimal_fitness = atr_gene._evaluate_individual([14])
        
        # Test suboptimal period
        suboptimal_fitness = atr_gene._evaluate_individual([7])
        
        assert optimal_fitness > suboptimal_fitness
    
    def test_trend_response(self, atr_gene, market_data):
        """Test ATR response to different market conditions"""
        high, low, close = market_data
        atr_gene.value = 14
        
        # Normal market
        normal_atr = atr_gene.compute(high, low, close)
        
        # Simulate trending market with increasing volatility
        trend = np.linspace(1.0, 1.5, len(high))  # Trend più forte
        trend_high = high.copy() * trend
        trend_low = low.copy() * (2.0 - trend)  # Movimento opposto per i minimi
        trend_atr = atr_gene.compute(trend_high, trend_low, close)
        
        # ATR should be higher in trending market
        assert trend_atr.mean() > normal_atr.mean()
    
    def test_invalid_inputs(self, atr_gene):
        """Test compute method with invalid inputs"""
        with pytest.raises(ValueError):
            atr_gene.compute(
                [1, 2, 3],  # Not a pandas Series
                pd.Series([1, 2, 3]),
                pd.Series([1, 2, 3])
            )
