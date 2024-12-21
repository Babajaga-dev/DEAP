import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.rsi_gene import RSIGene
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def mock_config_data():
    """Fixture that provides test configuration data"""
    return {
        "indicators": {
            "rsi": {
                "name": "Relative Strength Index",
                "period": {
                    "min": 2,
                    "max": 50,
                    "default": 14
                },
                "step": 1,
                "mutation_rate": 0.1,
                "mutation_range": 0.2
            }
        }
    }

@pytest.fixture
def mock_config_file(tmp_path, mock_config_data):
    """Fixture that creates a temporary config file"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "indicators.yaml"
    
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(mock_config_data, f)
    
    return config_file

@pytest.fixture
def config_loader(mock_config_file):
    """Fixture that provides a configured ConfigLoader"""
    loader = ConfigLoader()
    loader.config_dir = str(mock_config_file.parent)
    return loader

@pytest.fixture
def price_data():
    """Fixture that provides sample price data"""
    return pd.Series(
        [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0],
        index=pd.date_range(start='2024-01-01', periods=10, freq='D'),
        name='close'
    )

@pytest.fixture
def rsi_gene(config_loader):
    """Fixture that provides an RSI gene instance"""
    return RSIGene(config_loader)

class TestRSIGene:
    def test_initialization(self, rsi_gene):
        """Test RSI gene initialization"""
        assert isinstance(rsi_gene, RSIGene)
        assert rsi_gene.value is not None
        assert isinstance(rsi_gene.value, float)
        assert float(rsi_gene.value).is_integer()
    
    def test_compute(self, rsi_gene, price_data):
        """Test RSI computation"""
        rsi_gene.value = 14  # Standard RSI period
        result = rsi_gene.compute(price_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_data)
        
        # First n-1 values should be NaN
        assert pd.isna(result.iloc[0])
        
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 100).all()
    
    def test_compute_with_invalid_input(self, rsi_gene):
        """Test compute method with invalid input"""
        with pytest.raises(ValueError):
            rsi_gene.compute([1, 2, 3])  # Not a pandas Series
    
    def test_value_rounding(self, rsi_gene):
        """Test that period values are always rounded to integers"""
        rsi_gene.value = 14.7
        assert rsi_gene.value == 15.0
        assert float(rsi_gene.value).is_integer()
        
        rsi_gene.value = 14.2
        assert rsi_gene.value == 14.0
        assert float(rsi_gene.value).is_integer()
    
    def test_cache_functionality(self, rsi_gene, price_data):
        """Test that caching works correctly"""
        rsi_gene.value = 14
        
        # First computation
        result1 = rsi_gene.compute(price_data)
        
        # Second computation with same parameters
        result2 = rsi_gene.compute(price_data)
        
        # Should get same object from cache
        assert result1 is result2
    
    def test_crossover(self, rsi_gene, config_loader):
        """Test crossover operation"""
        parent2 = RSIGene(config_loader)
        
        rsi_gene.value = 10
        parent2.value = 20
        
        child1, child2 = rsi_gene.crossover(parent2)
        
        assert isinstance(child1, RSIGene)
        assert isinstance(child2, RSIGene)
        assert float(child1.value).is_integer()
        assert float(child2.value).is_integer()
        
        # Children values should be between parents
        min_val = min(rsi_gene.value, parent2.value)
        max_val = max(rsi_gene.value, parent2.value)
        assert min_val <= max(child1.value, child2.value)
        assert max_val >= min(child1.value, child2.value)
    
    def test_mutation(self, rsi_gene):
        """Test mutation operation"""
        original_value = rsi_gene.value
        mutated = False
        
        # Try multiple mutations as it's probabilistic
        for _ in range(10):
            rsi_gene.mutate()
            if rsi_gene.value != original_value:
                mutated = True
                break
        
        assert mutated, "Value should change after multiple mutations"
        assert isinstance(rsi_gene.value, float)
        assert float(rsi_gene.value).is_integer()
    
    def test_to_dict(self, rsi_gene):
        """Test dictionary representation"""
        rsi_gene.value = 14
        result = rsi_gene.to_dict()
        
        assert result["period"] == 14
        assert result["indicator_type"] == "RSI"
        assert result["library"] == "TA-Lib"
        assert result["range"] == "0-100"
    
    def test_evaluate_individual(self, rsi_gene):
        """Test the fitness evaluation"""
        # Test optimal period (14)
        optimal_fitness = rsi_gene._evaluate_individual([14])
        
        # Test suboptimal period
        suboptimal_fitness = rsi_gene._evaluate_individual([7])
        
        assert optimal_fitness > suboptimal_fitness
    
    def test_analyze_oversold_overbought(self, rsi_gene, price_data):
        """Test oversold/overbought analysis"""
        rsi_gene.value = 14
        signals = rsi_gene.analyze_oversold_overbought(
            price_data,
            oversold=30,
            overbought=70
        )
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(price_data)
        assert set(signals.unique()).issubset({-1, 0, 1})