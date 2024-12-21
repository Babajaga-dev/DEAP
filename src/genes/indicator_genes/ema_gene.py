from typing import Any, Tuple, Optional
import pandas as pd
import numpy as np
import talib
import random
from deap import tools, base, creator
from ..base_gene import BaseGene, GeneConfig
from ...utils.config_loader import ConfigLoader

class EMAGene(BaseGene):
    """Gene class for Exponential Moving Average indicator using TA-Lib and DEAP"""
    
    def __init__(self, config: Optional[GeneConfig] = None):
        """Initialize EMA gene with configuration"""
        if config is None:
            config_loader = ConfigLoader()
            indicator_config = config_loader.get_indicator_config("ema")
            config = GeneConfig(
                name=indicator_config["name"],
                min_value=indicator_config["min_period"],
                max_value=indicator_config["max_period"],
                step=indicator_config["step"],
                mutation_rate=indicator_config["mutation_rate"],
                mutation_sigma=indicator_config["mutation_range"]
            )
        super().__init__(config)
        self._cache = {}

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
        
        # Usa random.randint per generare valori interi
        self.toolbox.register("attr_int", lambda: float(random.randint(
            int(self.config.min_value), int(self.config.max_value)
        )))
        self.toolbox.register("individual", tools.initRepeat, 
                            getattr(creator, gene_type_name), 
                            self.toolbox.attr_int, n=1)
        self.toolbox.register("population", tools.initRepeat, 
                            list, self.toolbox.individual)
        
        # Usa operatori DEAP specifici per valori numerici
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._custom_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_individual)
    
    def _custom_mutation(self, individual):
        """Custom mutation operator that ensures integer values and changes"""
        current = int(individual[0])
        min_val = int(self.config.min_value)
        max_val = int(self.config.max_value)
        
        # Calcola un offset casuale che garantisce un cambio di valore
        offset = random.choice([-2, -1, 1, 2])
        new_value = current + offset
        
        # Assicura che il nuovo valore sia nell'intervallo valido
        if new_value < min_val:
            new_value = min_val + abs(offset)
        elif new_value > max_val:
            new_value = max_val - abs(offset)
        
        individual[0] = float(new_value)
        return individual,
    
    def compute(self, data: pd.Series) -> pd.Series:
        """
        Compute EMA for the given data using TA-Lib
        
        Args:
            data (pd.Series): Price data series
            
        Returns:
            pd.Series: Computed EMA values
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series")
        
        cache_key = (data.name, self.value)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        period = int(self.value)
        result = pd.Series(
            talib.EMA(data, timeperiod=period),
            index=data.index,
            name=f'EMA_{period}'
        )
        
        self._cache[cache_key] = result
        return result
    
    def _evaluate_individual(self, individual: list) -> Tuple[float]:
        """
        Evaluate the fitness of an individual (EMA period)
        Prefers common EMA periods (e.g., 9, 21, 50, 200)
        """
        period = int(individual[0])
        
        # Common EMA periods
        common_periods = [9, 21, 50, 200]
        
        # Find closest common period
        min_distance = min(abs(period - cp) for cp in common_periods)
        
        # Fitness based on proximity to common periods
        fitness = 1.0 / (1.0 + min_distance)
        
        return (fitness,)
    
    def validate_and_clip_value(self, value: float) -> float:
        """Ensure valid periods while maintaining float type"""
        clipped = super().validate_and_clip_value(value)
        return float(round(clipped))
    
    def crossover(self, other: 'EMAGene') -> tuple['EMAGene', 'EMAGene']:
        """Custom crossover that maintains period hierarchy if needed"""
        child1, child2 = super().crossover(other)
        return child1, child2
    
    def to_dict(self) -> dict:
        """Dictionary representation with EMA-specific information"""
        base_dict = super().to_dict()
        base_dict.update({
            "period": int(self.value),
            "indicator_type": "EMA",
            "library": "TA-Lib",
            "description": "Exponential Moving Average"
        })
        return base_dict
    
    @staticmethod
    def calculate_multiplier(period: int) -> float:
        """Calculate EMA multiplier for given period"""
        return 2.0 / (period + 1)
