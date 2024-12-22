from typing import Type, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd
from deap import base, creator, tools
from pathlib import Path
from ..strategies.base_strategy import BaseStrategy, StrategyConfig
from ..utils.config_loader import ConfigLoader
from ..data.data_loader import DataLoader

class StrategyWrapper:
    """Wrapper class for strategy to work with DEAP"""
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.fitness = creator.FitnessMax()

    def evaluate(self, data):
        return self.strategy.evaluate(data)

    def mutate(self):
        self.strategy.mutate()

    def crossover(self, other):
        s1, s2 = self.strategy.crossover(other.strategy)
        return StrategyWrapper(s1), StrategyWrapper(s2)

    @property
    def unwrap(self) -> BaseStrategy:
        """Get the wrapped strategy"""
        return self.strategy

class GeneticOptimizer:
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 config_loader: ConfigLoader,
                 population_size: int = 50,
                 generations: int = 100,
                 tournament_size: int = 3,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2):
        
        self.strategy_class = strategy_class
        self.config_loader = config_loader
        self.data_loader = DataLoader()
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Training data
        self.train_data = None
        self.test_data = None
        
        self._setup_deap()

    def load_data(self, 
                 file_path: Path,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 train_ratio: float = 0.8) -> None:
        """Load and prepare data for optimization"""
        self.data_loader.load_csv(file_path)
        
        if start_date or end_date:
            self.data_loader._data = self.data_loader.get_timeframe_data(start_date, end_date)
            
        self.train_data, self.test_data = self.data_loader.get_train_test_split(train_ratio=train_ratio)

    def _setup_deap(self):
        """Setup DEAP framework"""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_strategy)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
    
    def _create_individual(self) -> 'StrategyWrapper':
        """Create a new strategy instance"""
        strategy_type = self.strategy_class.__name__.lower()
        
        if strategy_type == 'mockstrategy':
            # Per MockStrategy usata nei test, usa configurazione di base
            strategy_params = {
                'name': 'MockStrategy',
                'risk_free_rate': 0.02,
                'transaction_costs': 0.001,
                'position_sizing': {'method': 'fixed', 'base_size': 1.0},
                'indicators': ['sma', 'rsi'],
                'entry_conditions': {},
                'exit_conditions': {},
                'risk_management': {},
                'option_preferences': {}
            }
        else:
            # Per strategie reali, usa la configurazione dal file
            strategies_config = self.config_loader.load_config("strategies")
            strategy_type = strategy_type.replace('strategy', '')
            
            if "strategies" not in strategies_config or strategy_type not in strategies_config["strategies"]:
                raise ValueError(f"Strategy type {strategy_type} not found in configuration")
                
            strategy_config = strategies_config["strategies"][strategy_type]
            general_config = strategies_config["strategies"].get('general', {})
            
            strategy_params = {
                'name': strategy_config.get('name', 'MockStrategy'),
                'risk_free_rate': general_config.get('risk_free_rate', 0.02),
                'transaction_costs': general_config.get('transaction_costs', 0.001),
                'position_sizing': strategy_config.get('position_sizing', {'method': 'fixed', 'base_size': 1.0}),
                'indicators': strategy_config.get('indicators', ['sma', 'rsi']),
                'entry_conditions': strategy_config.get('entry_conditions', {}),
                'exit_conditions': strategy_config.get('exit_conditions', {}),
                'risk_management': strategy_config.get('risk_management', {}),
                'option_preferences': strategy_config.get('option_preferences', {})
            }
        config = StrategyConfig(**strategy_params)
        strategy = self.strategy_class(config)
        return StrategyWrapper(strategy)
    
    def _evaluate_strategy(self, wrapper: 'StrategyWrapper') -> Tuple[float]:
        """
        Evaluate strategy fitness
        Args:
            wrapper: Strategy wrapper to evaluate
        Returns:
            Tuple[float]: Fitness value
        """
        if self.train_data is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        metrics = wrapper.evaluate(self.train_data)
        
        # Combine different metrics for fitness
        sharpe = metrics['sharpe_ratio']
        dd_penalty = abs(metrics['max_drawdown'])
        ret_bonus = max(0, metrics['annual_return'])  # Bonus for positive returns
        
        fitness = sharpe + ret_bonus - dd_penalty
        return (fitness,)
    
    def _crossover(self, wrapper1: 'StrategyWrapper', wrapper2: 'StrategyWrapper') -> Tuple['StrategyWrapper', 'StrategyWrapper']:
        """
        Perform crossover between two strategies
        Args:
            wrapper1: First strategy wrapper
            wrapper2: Second strategy wrapper
        Returns:
            Tuple: Two new strategy wrappers after crossover
        """
        if np.random.random() < self.crossover_prob:
            wrapper1, wrapper2 = wrapper1.crossover(wrapper2)
        return wrapper1, wrapper2
    
    def _mutate(self, wrapper: 'StrategyWrapper') -> Tuple['StrategyWrapper']:
        """
        Mutate strategy
        Args:
            wrapper: Strategy wrapper to mutate
        Returns:
            Tuple: Mutated strategy wrapper
        """
        if np.random.random() < self.mutation_prob:
            wrapper.mutate()
        return (wrapper,)
    
    def optimize(self, callback: Optional[Callable] = None) -> Tuple[BaseStrategy, List[dict]]:
        """Run genetic optimization"""
        if self.train_data is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        pop = self.toolbox.population(n=self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = tools.Logbook()
        
        # Evaluate initial population
        fitnesses = [self.toolbox.evaluate(ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        for gen in range(self.generations):
            # Select next generation
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for i in range(1, len(offspring), 2):
                if i + 1 < len(offspring):
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            # Apply mutation
            for i in range(len(offspring)):
                if np.random.random() < self.mutation_prob:
                    offspring[i] = self.toolbox.mutate(offspring[i])[0]
                    del offspring[i].fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.toolbox.evaluate(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
            
            # Record statistics
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
            
            if callback:
                callback(gen, record)
        
        best_ind = tools.selBest(pop, 1)[0]
        return best_ind.unwrap, logbook
    
    def validate(self, strategy: BaseStrategy) -> dict:
        """Validate strategy on test data"""
        if self.test_data is None:
            raise ValueError("No test data available. Call load_data first.")
            
        return strategy.evaluate(self.test_data)
