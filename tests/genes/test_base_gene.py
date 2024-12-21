import pytest
import pandas as pd
from deap import base, creator
from src.genes.base_gene import BaseGene, GeneConfig

# Concrete implementation of BaseGene for testing
# Definita fuori dalla classe di test per evitare problemi con pytest
class ConcreteGene(BaseGene):
    """A concrete implementation of BaseGene used only for testing"""
    def compute(self, data):
        return data * float(self.value)

@pytest.fixture
def gene_config():
    """Fixture that provides a basic gene configuration"""
    return GeneConfig(
        name="test_gene",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        mutation_rate=0.1,
        mutation_sigma=0.2
    )

@pytest.fixture
def test_gene(gene_config):
    """Fixture that provides a test gene instance"""
    return ConcreteGene(gene_config)

class TestBaseGene:
    def test_initialization(self, gene_config):
        """Test gene initialization"""
        gene = ConcreteGene(gene_config)
        assert gene.config == gene_config
        assert hasattr(gene, 'toolbox')
        assert gene._value is not None
    
    def test_value_property(self, test_gene):
        """Test value getter and setter"""
        test_gene.value = 0.5
        assert test_gene.value == 0.5
        
        # Test value clipping
        test_gene.value = 2.0
        assert test_gene.value == 1.0  # Clipped to max_value
        
        test_gene.value = -1.0
        assert test_gene.value == 0.0  # Clipped to min_value
    
    def test_randomize(self, test_gene):
        """Test random value initialization"""
        initial_value = test_gene.value
        test_gene.randomize()
        assert test_gene.value != initial_value
        assert 0.0 <= test_gene.value <= 1.0
    
    def test_mutate(self, test_gene):
        """Test mutation operation"""
        test_gene.value = 0.5
        initial_value = test_gene.value
        
        # Multiple mutations to ensure at least one change
        # (mutation is probabilistic)
        changed = False
        for _ in range(10):
            test_gene.mutate()
            if test_gene.value != initial_value:
                changed = True
                break
        
        assert changed, "Value should change after multiple mutations"
        assert 0.0 <= test_gene.value <= 1.0
    
    def test_crossover(self, gene_config):
        """Test crossover operation"""
        parent1 = ConcreteGene(gene_config)
        parent2 = ConcreteGene(gene_config)
        
        parent1.value = 0.2
        parent2.value = 0.8
        
        child1, child2 = parent1.crossover(parent2)
        
        assert isinstance(child1, ConcreteGene)
        assert isinstance(child2, ConcreteGene)
        assert 0.0 <= child1.value <= 1.0
        assert 0.0 <= child2.value <= 1.0
        
        # Children values should be between parents
        assert min(parent1.value, parent2.value) <= max(child1.value, child2.value)
        assert max(parent1.value, parent2.value) >= min(child1.value, child2.value)
    
    def test_invalid_crossover(self, test_gene, gene_config):
        """Test crossover with incompatible gene types"""
        class OtherGene(BaseGene):
            def compute(self, data): pass
        
        other_gene = OtherGene(gene_config)
        
        with pytest.raises(ValueError):
            test_gene.crossover(other_gene)
    
    def test_validate_and_clip_value(self, test_gene):
        """Test value validation and clipping"""
        assert test_gene.validate_and_clip_value(0.5) == 0.5
        assert test_gene.validate_and_clip_value(1.5) == 1.0
        assert test_gene.validate_and_clip_value(-0.5) == 0.0
    
    def test_to_dict(self, test_gene):
        """Test dictionary representation"""
        test_gene.value = 0.5
        gene_dict = test_gene.to_dict()
        
        assert gene_dict["name"] == "test_gene"
        assert gene_dict["value"] == 0.5
        assert gene_dict["type"] == "ConcreteGene"
    
    def test_compute(self, test_gene):
        """Test compute method"""
        test_gene.value = 0.5
        result = test_gene.compute(2.0)
        assert result == 1.0  # 2.0 * 0.5
    
    def test_invalid_config(self):
        """Test initialization with invalid configuration"""
        with pytest.raises(ValueError):
            GeneConfig(
                name="invalid",
                min_value=1.0,
                max_value=0.0,  # Invalid: min > max
                step=0.1,
                mutation_rate=0.1,
                mutation_sigma=0.2
            )
        
        with pytest.raises(ValueError):
            GeneConfig(
                name="invalid",
                min_value=0.0,
                max_value=1.0,
                step=-0.1,  # Invalid: negative step
                mutation_rate=0.1,
                mutation_sigma=0.2
            )
