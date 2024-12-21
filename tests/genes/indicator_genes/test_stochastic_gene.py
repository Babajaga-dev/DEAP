import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.stochastic_gene import StochasticGene

@pytest.fixture
def market_data():
    """Fixture that provides sample market data including high, low, close prices"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Create data that will produce extreme stochastic values
    # First create a downtrend to get oversold conditions
    close = pd.Series(data=[100, 98, 96, 94, 92, 90, 88, 86, 84, 82,
                           # Then create an uptrend to get overbought conditions
                           84, 86, 88, 90, 92, 94, 96, 98, 100, 102,
                           # Then some consolidation
                           101, 102, 101, 102, 101, 102, 101, 102, 101, 102], 
                     index=dates, name='close')
    
    # High and low with wider range to ensure extreme stochastic values
    high = close + 2.0
    low = close - 2.0
    
    return close, high, low

@pytest.fixture
def stochastic_gene():
    """Fixture that provides a Stochastic Oscillator gene instance"""
    return StochasticGene()

class TestStochasticGene:
    def test_initialization(self):
        """Test Stochastic gene initialization"""
        gene = StochasticGene()
        assert isinstance(gene, StochasticGene)
        assert hasattr(gene, 'fastk_period')
        assert hasattr(gene, 'slowk_period')
        assert hasattr(gene, 'slowd_period')
        
        # Verify default periods are within valid ranges
        assert 5 <= gene.fastk_period <= 30
        assert 1 <= gene.slowk_period <= 10
        assert 1 <= gene.slowd_period <= 10
    
    def test_compute(self, stochastic_gene, market_data):
        """Test Stochastic computation"""
        close, high, low = market_data
        stochastic_gene.value = [14, 3, 3]  # Standard parameters
        
        slowk, slowd = stochastic_gene.compute(close, high, low)
        
        assert isinstance(slowk, pd.Series)
        assert isinstance(slowd, pd.Series)
        assert len(slowk) == len(close)
        
        # First few values should be NaN
        assert pd.isna(slowk[:13]).all()  # fastk_period-1 values
        assert pd.isna(slowd[:15]).all()  # Additional slowd_period-1 values
        
        # Test value ranges
        valid_k = slowk.dropna()
        valid_d = slowd.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    def test_param_validation(self, stochastic_gene):
        """Test parameter validation"""
        # Test minimum periods
        with pytest.raises(ValueError):
            stochastic_gene.value = [4, 3, 3]  # fastk_period too low
            
        with pytest.raises(ValueError):
            stochastic_gene.value = [14, 0, 3]  # slowk_period too low
            
        with pytest.raises(ValueError):
            stochastic_gene.value = [14, 3, 0]  # slowd_period too low
    
    def test_crossover(self, stochastic_gene):
        """Test crossover operation"""
        parent2 = StochasticGene()
        
        stochastic_gene.value = [14, 3, 3]
        parent2.value = [10, 2, 2]
        
        child1, child2 = stochastic_gene.crossover(parent2)
        
        for child in [child1, child2]:
            assert isinstance(child, StochasticGene)
            assert 5 <= child.fastk_period <= 30
            assert 1 <= child.slowk_period <= 10
            assert 1 <= child.slowd_period <= 10
    
    def test_mutation(self, stochastic_gene):
        """Test mutation operation"""
        original_values = [
            stochastic_gene.fastk_period,
            stochastic_gene.slowk_period,
            stochastic_gene.slowd_period
        ]
        
        mutated = False
        for _ in range(10):
            stochastic_gene.mutate()
            current_values = [
                stochastic_gene.fastk_period,
                stochastic_gene.slowk_period,
                stochastic_gene.slowd_period
            ]
            if current_values != original_values:
                mutated = True
                break
        
        assert mutated, "Values should change after multiple mutations"
        assert 5 <= stochastic_gene.fastk_period <= 30
        assert 1 <= stochastic_gene.slowk_period <= 10
        assert 1 <= stochastic_gene.slowd_period <= 10
    
    def test_overbought_oversold(self, stochastic_gene, market_data):
        """Test overbought/oversold identification"""
        close, high, low = market_data
        stochastic_gene.value = [14, 3, 3]
        
        slowk, slowd = stochastic_gene.compute(close, high, low)
        
        # Test if values correctly identify overbought (>80) and oversold (<20) conditions
        valid_values = slowk.dropna()
        assert ((valid_values >= 0) & (valid_values <= 20) | 
                (valid_values >= 80) & (valid_values <= 100)).any()
    
    def test_cache_functionality(self, stochastic_gene, market_data):
        """Test that caching works correctly"""
        close, high, low = market_data
        stochastic_gene.value = [14, 3, 3]
        
        # First computation
        result1 = stochastic_gene.compute(close, high, low)
        
        # Second computation with same parameters
        result2 = stochastic_gene.compute(close, high, low)
        
        # Should get same objects from cache
        assert result1[0] is result2[0]  # slowk
        assert result1[1] is result2[1]  # slowd
    
    def test_to_dict(self, stochastic_gene):
        """Test dictionary representation"""
        stochastic_gene.value = [14, 3, 3]
        result = stochastic_gene.to_dict()
        
        assert result["fastk_period"] == 14
        assert result["slowk_period"] == 3
        assert result["slowd_period"] == 3
        assert result["indicator_type"] == "Stochastic"
        assert result["library"] == "TA-Lib"
        assert result["range"] == "0-100"
    
    def test_evaluate_individual(self, stochastic_gene):
        """Test the fitness evaluation"""
        # Test standard parameters (14,3,3)
        optimal_fitness = stochastic_gene._evaluate_individual([14, 3, 3])
        
        # Test non-standard parameters
        suboptimal_fitness = stochastic_gene._evaluate_individual([10, 2, 2])
        
        assert optimal_fitness > suboptimal_fitness
    
    def test_invalid_inputs(self, stochastic_gene):
        """Test compute method with invalid inputs"""
        with pytest.raises(ValueError):
            stochastic_gene.compute(
                [1, 2, 3],  # Not a pandas Series
                pd.Series([1, 2, 3]),
                pd.Series([1, 2, 3])
            )
