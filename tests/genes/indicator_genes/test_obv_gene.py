import pytest
import pandas as pd
import numpy as np
from src.genes.indicator_genes.obv_gene import OBVGene

@pytest.fixture
def market_data():
    """Fixture that provides sample market data including price and volume"""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    # Create price data with clear up/down trends
    base_price = 100
    prices = []
    volumes = []
    
    # Simulate price movements with corresponding volume patterns
    for i in range(50):
        if i < 25:  # Uptrend
            base_price *= 1.01  # 1% increase
            volume = np.random.normal(1000000, 100000)  # Higher volume on uptrend
        else:  # Downtrend
            base_price *= 0.99  # 1% decrease
            volume = np.random.normal(800000, 100000)  # Lower volume on downtrend
        
        prices.append(base_price)
        volumes.append(max(volume, 0))  # Ensure positive volume
    
    return pd.Series(prices, index=dates, name='close'), pd.Series(volumes, index=dates, name='volume')

@pytest.fixture
def obv_gene():
    """Fixture that provides an OBV gene instance"""
    return OBVGene()

class TestOBVGene:
    def test_initialization(self):
        """Test OBV gene initialization"""
        gene = OBVGene()
        assert isinstance(gene, OBVGene)
        assert gene.value is not None
        assert isinstance(gene.value, float)
        assert gene.value.is_integer()
    
    def test_compute(self, obv_gene, market_data):
        """Test OBV computation"""
        close, volume = market_data
        obv_gene.value = 20  # Signal line period
        
        obv, signal = obv_gene.compute(close, volume)
        
        assert isinstance(obv, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(obv) == len(close)
        assert len(signal) == len(close)
        
        # First values of signal line should be NaN
        assert pd.isna(signal[:19]).all()
        
        # OBV should be cumulative and reflect volume
        assert not pd.isna(obv.iloc[0])  # First OBV value should be valid
        
        # Test if OBV increases on up days and decreases on down days
        price_changes = close.diff()
        obv_changes = obv.diff()
        
        up_days = price_changes > 0
        assert (obv_changes[up_days] >= 0).all()
        
        down_days = price_changes < 0
        assert (obv_changes[down_days] <= 0).all()
    
    def test_compute_trigger(self, obv_gene, market_data):
        """Test trading trigger computation"""
        close, volume = market_data
        obv_gene.value = 20
        
        triggers = obv_gene.compute_trigger(close, volume)
        
        assert isinstance(triggers, pd.Series)
        assert len(triggers) == len(close)
        assert set(triggers.dropna().unique()).issubset({-1, 0, 1})
    
    def test_analyze_divergence(self, obv_gene, market_data):
        """Test divergence analysis"""
        close, volume = market_data
        
        divergence = obv_gene.analyze_divergence(close, volume)
        
        assert isinstance(divergence, pd.Series)
        assert len(divergence) == len(close)
        assert set(divergence.unique()).issubset({-1, 0, 1})
    
    def test_crossover(self, obv_gene):
        """Test crossover operation"""
        parent2 = OBVGene()
        
        obv_gene.value = 20
        parent2.value = 30
        
        child1, child2 = obv_gene.crossover(parent2)
        
        for child in [child1, child2]:
            assert isinstance(child, OBVGene)
            assert isinstance(child.value, float)
            assert child.value.is_integer()
            assert child.config.min_value <= child.value <= child.config.max_value
    
    def test_mutation(self, obv_gene):
        """Test mutation operation"""
        original_value = obv_gene.value
        
        mutated = False
        for _ in range(10):
            obv_gene.mutate()
            if obv_gene.value != original_value:
                mutated = True
                break
        
        assert mutated, "Value should change after multiple mutations"
        assert isinstance(obv_gene.value, float)
        assert obv_gene.value.is_integer()
    
    def test_cache_functionality(self, obv_gene, market_data):
        """Test that caching works correctly"""
        close, volume = market_data
        obv_gene.value = 20
        
        # First computation
        result1 = obv_gene.compute(close, volume)
        
        # Second computation with same parameters
        result2 = obv_gene.compute(close, volume)
        
        # Should get same objects from cache
        assert result1[0] is result2[0]  # OBV line
        assert result1[1] is result2[1]  # Signal line
    
    def test_to_dict(self, obv_gene):
        """Test dictionary representation"""
        obv_gene.value = 20
        result = obv_gene.to_dict()
        
        assert result["signal_period"] == 20
        assert result["indicator_type"] == "OBV"
        assert result["library"] == "TA-Lib"
        assert "description" in result
    
    def test_evaluate_individual(self, obv_gene):
        """Test the fitness evaluation"""
        # Test standard signal period (20)
        optimal_fitness = obv_gene._evaluate_individual([20])
        
        # Test non-standard period
        suboptimal_fitness = obv_gene._evaluate_individual([13])
        
        assert optimal_fitness > suboptimal_fitness
    
    def test_trend_confirmation(self, obv_gene):
        """Test OBV trend confirmation characteristics"""
        # Create data with strong trend and corresponding volume
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Uptrend with increasing volume
        prices = np.linspace(100, 200, 50)  # Strong uptrend
        volumes = np.linspace(1000000, 2000000, 50)  # Increasing volume
        
        close = pd.Series(prices, index=dates)
        volume = pd.Series(volumes, index=dates)
        
        obv, signal = obv_gene.compute(close, volume)
        
        # In a strong uptrend with increasing volume, OBV should trend up
        assert obv.iloc[-1] > obv.iloc[0]
    
    def test_invalid_inputs(self, obv_gene):
        """Test compute method with invalid inputs"""
        with pytest.raises(ValueError):
            obv_gene.compute(
                [1, 2, 3],  # Not a pandas Series
                pd.Series([1, 2, 3])
            )
