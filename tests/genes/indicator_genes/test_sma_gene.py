import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.sma_gene import SMAGene

@pytest.fixture
def price_data():
    """Fixture that provides sample price data"""
    return pd.Series(
        [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0],
        index=pd.date_range(start='2024-01-01', periods=10, freq='D'),
        name='close'
    )

@pytest.fixture
def sma_gene():
    """Fixture that provides an SMA gene instance"""
    return SMAGene()

class TestSMAGene:
    def test_initialization(self):
        """Test SMA gene initialization"""
        gene = SMAGene()
        assert isinstance(gene, SMAGene)
        assert gene.value is not None
        assert isinstance(gene.value, float)
    
    def test_compute(self, sma_gene, price_data):
        """Test SMA computation"""
        sma_gene.value = 3  # Set period to 3
        result = sma_gene.compute(price_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_data)
        assert pd.isna(result[0])  # First two values should be NaN
        assert pd.isna(result[1])  # because period is 3
        
        # Test some specific values
        expected_value = (101.0 + 102.0 + 103.0) / 3
        assert np.isclose(result[3], expected_value)
    
    def test_compute_with_invalid_input(self, sma_gene):
        """Test compute method with invalid input"""
        with pytest.raises(ValueError):
            sma_gene.compute([1, 2, 3])  # Not a pandas Series
    
    def test_value_rounding(self, sma_gene):
        """Test that period values are always rounded to integers"""
        sma_gene.value = 3.7
        assert sma_gene.value == 4
        
        sma_gene.value = 3.2
        assert sma_gene.value == 3
    
    def test_cache_functionality(self, sma_gene, price_data):
        """Test that caching works correctly"""
        sma_gene.value = 3
        
        # First computation
        result1 = sma_gene.compute(price_data)
        
        # Second computation with same parameters
        result2 = sma_gene.compute(price_data)
        
        # Should get same object from cache
        assert result1 is result2
    
    def test_crossover(self):
        """Test crossover operation"""
        parent1 = SMAGene()
        parent2 = SMAGene()
        
        parent1.value = 5
        parent2.value = 15
        
        child1, child2 = parent1.crossover(parent2)
        
        # Check that children are valid SMAGene instances
        assert isinstance(child1, SMAGene)
        assert isinstance(child2, SMAGene)
        
        # Check that children have integer periods
        assert isinstance(child1.value, float) and child1.value.is_integer()
        assert isinstance(child2.value, float) and child2.value.is_integer()
        
        # Check that children values are between parents
        assert min(parent1.value, parent2.value) <= max(child1.value, child2.value)
        assert max(parent1.value, parent2.value) >= min(child1.value, child2.value)
    
    def test_mutation(self, sma_gene):
        """Test mutation operation"""
        original_value = sma_gene.value
        mutated = False
        
        # Try multiple mutations as it's probabilistic
        for _ in range(10):
            sma_gene.mutate()
            if sma_gene.value != original_value:
                mutated = True
                break
        
        assert mutated, "Value should change after multiple mutations"
        assert isinstance(sma_gene.value, float)
        assert sma_gene.value.is_integer()
    
    def test_to_dict(self, sma_gene):
        """Test dictionary representation"""
        sma_gene.value = 10
        result = sma_gene.to_dict()
        
        assert result["period"] == 10
        assert result["indicator_type"] == "SMA"
        assert result["library"] == "TA-Lib"
    
    def test_evaluate_individual(self, sma_gene):
        """Test the fitness evaluation"""
        fitness = sma_gene._evaluate_individual([10])
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert 0 <= fitness[0] <= 1