from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from deap import base, creator, tools
import random

@dataclass
class GeneConfig:
    """Configuration class for gene parameters"""
    name: str
    min_value: float
    max_value: float
    step: float = 0.1
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.2
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        if self.step <= 0:
            raise ValueError("step must be positive")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.mutation_sigma <= 0:
            raise ValueError("mutation_sigma must be positive")

class BaseGene(ABC):
    """Abstract base class for all genes in the trading system using DEAP"""
    
    def __init__(self, config: GeneConfig):
        """
        Initialize the base gene with configuration parameters
        
        Args:
            config (GeneConfig): Configuration parameters for the gene
        """
        self.config = config
        self.validate_config()
        
        # Register gene in DEAP
        self._register_gene()
        
        # Initialize with random value
        self.randomize()
    
    def _register_gene(self):
        """Register the gene type and required operators in DEAP"""
        # Create a new Fitness class if not already created
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create a gene class specific to this type if not already created
        gene_type_name = f"{self.__class__.__name__}Class"
        if not hasattr(creator, gene_type_name):
            creator.create(gene_type_name, list, fitness=creator.FitnessMax)
        
        # Register genetic operators in the toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 
                            self.config.min_value, self.config.max_value)
        self.toolbox.register("individual", tools.initRepeat, 
                            getattr(creator, gene_type_name), 
                            self.toolbox.attr_float, n=1)
        self.toolbox.register("population", tools.initRepeat, 
                            list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, 
                            mu=0, sigma=self.config.mutation_sigma, 
                            indpb=self.config.mutation_rate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_individual)
    
    def validate_config(self) -> None:
        """Validate that configuration exists"""
        if not self.config:
            raise ValueError("Gene configuration is required")
    
    @property
    def value(self) -> float:
        """Get the current value of the gene"""
        return self._value[0] if isinstance(self._value, list) else self._value
    
    @value.setter
    def value(self, new_value: float) -> None:
        """Set the value of the gene with validation"""
        if isinstance(new_value, list):
            self._value = [self.validate_and_clip_value(new_value[0])]
        else:
            self._value = [self.validate_and_clip_value(new_value)]
    
    def randomize(self) -> None:
        """Initialize the gene with a random valid value using DEAP"""
        individual = self.toolbox.individual()
        self._value = individual
    
    def crossover(self, other: 'BaseGene') -> tuple['BaseGene', 'BaseGene']:
        """Perform crossover with another gene using DEAP"""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot crossover with gene of different type: {type(other)}")
        
        child1, child2 = map(self.__class__, [self.config, self.config])
        offspring1, offspring2 = self.toolbox.mate(self._value, other._value)
        
        # Validate and clip the offspring values
        child1._value = [self.validate_and_clip_value(offspring1[0])]
        child2._value = [self.validate_and_clip_value(offspring2[0])]
        
        return child1, child2
    
    def mutate(self) -> None:
        """Mutate the gene using DEAP's mutation operator"""
        self._value, = self.toolbox.mutate(self._value)
        self._value = [self.validate_and_clip_value(self._value[0])]
    
    def validate_and_clip_value(self, value: float) -> float:
        """Validate and clip the value to be within the allowed range"""
        return max(min(value, self.config.max_value), self.config.min_value)
    
    def _evaluate_individual(self, individual):
        """
        Basic evaluation function. Override in specific genes if needed.
        Returns a tuple as required by DEAP.
        """
        return (0.0,)  # Default fitness of 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the gene to a dictionary representation"""
        return {
            "name": self.config.name,
            "value": self.value,
            "type": self.__class__.__name__
        }
    
    @abstractmethod
    def compute(self, data: Any) -> Any:
        """
        Compute the gene's output based on input data
        Must be implemented by specific gene classes
        """
        pass
