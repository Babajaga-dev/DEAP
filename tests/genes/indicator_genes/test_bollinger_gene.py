import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.bollinger_gene import BollingerGene

@pytest.fixture
def price_data():
    """Fixture that provides sample price data"""
    return pd.Series(
        [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0] * 2,
        index=pd.date_range(start='2024-01-01', periods=20, freq='D'),
        name='close'
    )

@pytest.fixture
def bollinger_gene():
    """Fixture that provides a Bollinger Bands gene instance"""
    return BollingerGene()

class TestBollingerGene:
    def test_initialization(self):
        """Test Bollinger Bands gene initialization"""
        gene = BollingerGene()
        assert isinstance(gene, BollingerGene)
        assert hasattr(gene, 'period')
        assert hasattr(gene, 'num_std')
        
        # Verify default values are within reasonable ranges
        assert 5 <= gene.period <= 50
        assert 1.0 <= gene.num_std <= 3.0
    
    def test_compute(self, bollinger_gene, price_data):
        """Test Bollinger Bands computation"""
        bollinger_gene.period = 20
        bollinger_gene.num_std = 2.0
        
        upper, middle, lower = bollinger_gene.compute(price_data)
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(price_data)
        
        # First few values should be NaN
        assert pd.isna(upper[:19]).all()
        
        # Test band relationships
        valid_indices = ~pd.isna(upper)
        assert (upper[valid_indices] > middle[valid_indices]).all()
        assert (lower[valid_indices] < middle[valid_indices]).all()
        
        # Test band symmetry
        np.testing.assert_array_almost_equal(
            (upper - middle)[valid_indices],
            (middle - lower)[valid_indices]
        )
    
    def test_band_width_with_std(self, bollinger_gene, price_data):
        """Test band width changes with different standard deviations"""
        bollinger_gene.period = 20
        
        # Test with 1 standard deviation
        bollinger_gene.num_std = 1.0
        upper1, middle1, lower1 = bollinger_gene.compute(price_data)
        width1 = upper1 - lower1
        
        # Test with 2 standard deviations
        bollinger_gene.num_std = 2.0
        upper2, middle2, lower2 = bollinger_gene.compute(price_data)
        width2 = upper2 - lower2
        
        # Width should double
        valid_indices = ~pd.isna(width1)
        np.testing.assert_array_almost_equal(
            width2[valid_indices],
            width1[valid_indices] * 2,
            decimal=2
        )
    
    def test_crossover(self, bollinger_gene):
        """Test crossover operation"""
        parent2 = BollingerGene()
        
        bollinger_gene.value = [20, 2.0]
        parent2.value = [10, 1.5]
        
        child1, child2 = bollinger_gene.crossover(parent2)
        
        for child in [child1, child2]:
            assert isinstance(child, BollingerGene)
            assert isinstance(child.period, (int, float))
            assert isinstance(child.num_std, float)
            assert 1.0 <= child.num_std <= 3.0
    
    def test_mutation(self, bollinger_gene):
        """Test mutation operation"""
        original_values = [bollinger_gene.period, bollinger_gene.num_std]
        
        mutated = False
        for _ in range(10):
            bollinger_gene.mutate()
            current_values = [bollinger_gene.period, bollinger_gene.num_std]
            if current_values != original_values:
                mutated = True
                break
        
        assert mutated, "Values should change after multiple mutations"
        assert 1.0 <= bollinger_gene.num_std <= 3.0
    
    def test_to_dict(self, bollinger_gene):
        """Test dictionary representation"""
        bollinger_gene.value = [20, 2.0]
        result = bollinger_gene.to_dict()
        
        assert result["period"] == 20
        assert result["standard_deviations"] == 2.0
        assert result["indicator_type"] == "Bollinger Bands"
        assert result["library"] == "TA-Lib"
    
    def test_cache_functionality(self, bollinger_gene, price_data):
        """Test that caching works correctly"""
        bollinger_gene.value = [20, 2.0]
        
        # First computation
        result1 = bollinger_gene.compute(price_data)
        
        # Second computation with same parameters
        result2 = bollinger_gene.compute(price_data)
        
        # Should get same objects from cache
        assert result1[0] is result2[0]  # Upper band
        assert result1[1] is result2[1]  # Middle band
        assert result1[2] is result2[2]  # Lower band
    
    def test_evaluate_individual(self, bollinger_gene):
        """Test the fitness evaluation"""
        # Test standard parameters (20, 2.0)
        optimal_fitness = bollinger_gene._evaluate_individual([20, 2.0])
        
        # Test non-standard parameters
        suboptimal_fitness = bollinger_gene._evaluate_individual([10, 1.5])
        
        assert optimal_fitness > suboptimal_fitness
    
    def test_invalid_input(self, bollinger_gene):
        """Test compute method with invalid input"""
        with pytest.raises(ValueError):
            bollinger_gene.compute([1, 2, 3])  # Not a pandas Series