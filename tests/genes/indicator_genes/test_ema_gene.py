import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.ema_gene import EMAGene

@pytest.fixture
def price_data():
    """Fixture that provides sample price data"""
    # Create price data with a clear trend for testing EMA behavior
    base = np.linspace(100, 200, 50)  # Upward trend
    noise = np.random.normal(0, 2, 50)  # Add some noise
    prices = base + noise
    
    return pd.Series(
        prices,
        index=pd.date_range(start='2024-01-01', periods=50, freq='D'),
        name='close'
    )

@pytest.fixture
def ema_gene():
    """Fixture that provides an EMA gene instance"""
    return EMAGene()

class TestEMAGene:
    def test_initialization(self):
        """Test EMA gene initialization"""
        gene = EMAGene()
        assert isinstance(gene, EMAGene)
        assert gene.value is not None
        assert isinstance(gene.value, float)
        assert gene.value.is_integer()
        
        # Check if common periods are available
        assert hasattr(gene, 'calculate_multiplier')
    
    def test_compute(self, ema_gene, price_data):
        """Test EMA computation"""
        ema_gene.value = 20  # Standard EMA period
        
        result = ema_gene.compute(price_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_data)
        
        # First values should be NaN
        assert pd.isna(result[:19]).all()
        
        # Test that EMA follows the trend but lags behind
        valid_values = result.dropna()
        price_diff = price_data[valid_values.index].diff()
        ema_diff = valid_values.diff()
        
        # Rimuovi i valori NaN e assicurati che le serie abbiano la stessa lunghezza
        mask = ~(pd.isna(price_diff) | pd.isna(ema_diff))
        price_diff = price_diff[mask]
        ema_diff = ema_diff[mask]
        
        # When price moves up, EMA should move up (with lag)
        assert np.corrcoef(price_diff, ema_diff)[0, 1] > 0
    
    def test_different_periods(self, ema_gene, price_data):
        """Test EMA behavior with different periods"""
        # Compare short and long period EMAs
        ema_gene.value = 10
        short_ema = ema_gene.compute(price_data)
        
        ema_gene.value = 30
        long_ema = ema_gene.compute(price_data)
        
        # Short EMA should be more responsive (higher standard deviation)
        assert short_ema.std() > long_ema.std()
    
    def test_calculate_multiplier(self, ema_gene):
        """Test EMA multiplier calculation"""
        multiplier_10 = ema_gene.calculate_multiplier(10)
        multiplier_20 = ema_gene.calculate_multiplier(20)
        
        assert multiplier_10 > multiplier_20  # Shorter period = higher multiplier
        assert 0 < multiplier_10 < 1
        assert 0 < multiplier_20 < 1
    
    def test_crossover(self, ema_gene):
        """Test crossover operation"""
        parent2 = EMAGene()
        
        ema_gene.value = 20
        parent2.value = 50
        
        child1, child2 = ema_gene.crossover(parent2)
        
        for child in [child1, child2]:
            assert isinstance(child, EMAGene)
            assert isinstance(child.value, float)
            assert child.value.is_integer()
            assert child.config.min_value <= child.value <= child.config.max_value
    
    def test_mutation(self, ema_gene):
        """Test mutation operation"""
        original_value = ema_gene.value
        
        mutated = False
        for _ in range(10):
            ema_gene.mutate()
            if ema_gene.value != original_value:
                mutated = True
                break
        
        assert mutated, "Value should change after multiple mutations"
        assert isinstance(ema_gene.value, float)
        assert ema_gene.value.is_integer()
    
    def test_cache_functionality(self, ema_gene, price_data):
        """Test that caching works correctly"""
        ema_gene.value = 20
        
        # First computation
        result1 = ema_gene.compute(price_data)
        
        # Second computation with same parameters
        result2 = ema_gene.compute(price_data)
        
        # Should get same object from cache
        assert result1 is result2
    
    def test_to_dict(self, ema_gene):
        """Test dictionary representation"""
        ema_gene.value = 20
        result = ema_gene.to_dict()
        
        assert result["period"] == 20
        assert result["indicator_type"] == "EMA"
        assert result["library"] == "TA-Lib"
        assert "description" in result
    
    def test_evaluate_individual(self, ema_gene):
        """Test the fitness evaluation with common EMA periods"""
        # Test common periods (e.g., 9, 21, 50, 200)
        common_period_fitness = ema_gene._evaluate_individual([21])
        
        # Test uncommon period
        uncommon_period_fitness = ema_gene._evaluate_individual([23])
        
        assert common_period_fitness > uncommon_period_fitness
    
    def test_trend_following(self, ema_gene):
        """Test EMA trend following characteristics"""
        # Create strongly trending data
        trend_data = pd.Series(
            np.linspace(100, 200, 100),  # Strong uptrend
            index=pd.date_range(start='2024-01-01', periods=100, freq='D')
        )
        
        ema_gene.value = 20
        ema = ema_gene.compute(trend_data)
        
        # In a strong trend, EMA should follow closely
        correlation = np.corrcoef(
            trend_data[20:],  # Skip NaN values
            ema[20:]
        )[0, 1]
        
        assert correlation > 0.95  # Strong positive correlation
    
    def test_invalid_inputs(self, ema_gene):
        """Test compute method with invalid inputs"""
        with pytest.raises(ValueError):
            ema_gene.compute([1, 2, 3])  # Not a pandas Series
    
    def test_period_validation(self, ema_gene):
        """Test validation of EMA period"""
        config = ema_gene.config
        
        # Test minimum period
        ema_gene.value = config.min_value - 1
        assert ema_gene.value == config.min_value
        
        # Test maximum period
        ema_gene.value = config.max_value + 1
        assert ema_gene.value == config.max_value
