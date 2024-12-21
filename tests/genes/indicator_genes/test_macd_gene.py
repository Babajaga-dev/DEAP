import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.macd_gene import MACDGene

@pytest.fixture
def price_data():
    """Fixture that provides sample price data"""
    return pd.Series(
        [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0] * 3,  # Extended data for MACD
        index=pd.date_range(start='2024-01-01', periods=30, freq='D'),
        name='close'
    )

@pytest.fixture
def macd_gene():
    """Fixture that provides a MACD gene instance"""
    return MACDGene()

class TestMACDGene:
    def test_initialization(self):
        """Test MACD gene initialization"""
        gene = MACDGene()
        assert isinstance(gene, MACDGene)
        assert hasattr(gene, 'fast_period')
        assert hasattr(gene, 'slow_period')
        assert hasattr(gene, 'signal_period')
        
        # Verify default periods
        assert gene.fast_period < gene.slow_period
    
    def test_compute(self, macd_gene, price_data):
        """Test MACD computation"""
        macd_gene.fast_period = 12
        macd_gene.slow_period = 26
        macd_gene.signal_period = 9
        
        macd, signal, hist = macd_gene.compute(price_data)
        
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(price_data)
        
        # First few values should be NaN due to periods
        assert pd.isna(macd[:11]).all()  # First fast_period-1 values
        assert pd.isna(signal[:8]).all()  # Additional signal_period-1 values
        
        # Histogram should be MACD - Signal
        np.testing.assert_array_almost_equal(
            hist.dropna(),
            (macd - signal).dropna()
        )
    
    def test_period_validation(self, macd_gene):
        """Test validation of MACD periods"""
        # Test fast < slow period constraint
        macd_gene.value = [20, 15, 9]  # Invalid: fast > slow
        assert macd_gene.fast_period < macd_gene.slow_period
        
        # Test minimum periods
        config = macd_gene.config
        macd_gene.value = [
            config.min_value - 1,
            config.min_value * 2,
            config.min_value
        ]
        assert macd_gene.fast_period >= config.min_value
    
    def test_crossover(self, macd_gene):
        """Test crossover operation"""
        parent2 = MACDGene()
        
        macd_gene.value = [12, 26, 9]
        parent2.value = [10, 20, 7]
        
        child1, child2 = macd_gene.crossover(parent2)
        
        for child in [child1, child2]:
            assert isinstance(child, MACDGene)
            assert child.fast_period < child.slow_period
            assert isinstance(child.fast_period, (int, float))
            assert isinstance(child.slow_period, (int, float))
            assert isinstance(child.signal_period, (int, float))
    
    def test_mutation(self, macd_gene):
        """Test mutation operation"""
        original_values = [
            macd_gene.fast_period,
            macd_gene.slow_period,
            macd_gene.signal_period
        ]
        
        mutated = False
        for _ in range(10):
            macd_gene.mutate()
            current_values = [
                macd_gene.fast_period,
                macd_gene.slow_period,
                macd_gene.signal_period
            ]
            if current_values != original_values:
                mutated = True
                break
        
        assert mutated, "Values should change after multiple mutations"
        assert macd_gene.fast_period < macd_gene.slow_period
    
    def test_to_dict(self, macd_gene):
        """Test dictionary representation"""
        macd_gene.value = [12, 26, 9]
        result = macd_gene.to_dict()
        
        assert result["fast_period"] == 12
        assert result["slow_period"] == 26
        assert result["signal_period"] == 9
        assert result["indicator_type"] == "MACD"
        assert result["library"] == "TA-Lib"
    
    def test_cache_functionality(self, macd_gene, price_data):
        """Test that caching works correctly"""
        macd_gene.value = [12, 26, 9]
        
        # First computation
        result1 = macd_gene.compute(price_data)
        
        # Second computation with same parameters
        result2 = macd_gene.compute(price_data)
        
        # Should get same objects from cache
        assert result1[0] is result2[0]  # MACD line
        assert result1[1] is result2[1]  # Signal line
        assert result1[2] is result2[2]  # Histogram
    
    def test_evaluate_individual(self, macd_gene):
        """Test the fitness evaluation"""
        # Test standard MACD parameters (12,26,9)
        optimal_fitness = macd_gene._evaluate_individual([12, 26, 9])
        
        # Test non-standard parameters
        suboptimal_fitness = macd_gene._evaluate_individual([10, 20, 7])
        
        assert optimal_fitness > suboptimal_fitness
    
    def test_invalid_input(self, macd_gene):
        """Test compute method with invalid input"""
        with pytest.raises(ValueError):
            macd_gene.compute([1, 2, 3])  # Not a pandas Series