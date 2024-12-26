import numpy as np
import random
import logging
import copy
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
from deap import base, creator, tools
from ..data.data_loader import DataLoader

logger = logging.getLogger(__name__)

def clear_deap():
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

class Individual:
    """Classe Individual personalizzata per DEAP"""
    def __init__(self, strategy=None):
        self.strategy = strategy
        self.fitness = creator.FitnessMax()
        self.fitness.values = (0.0,)  # Inizializza con un valore di default

    def __deepcopy__(self, memo):
        result = Individual()
        memo[id(self)] = result
        result.strategy = copy.deepcopy(self.strategy, memo)
        result.fitness = copy.deepcopy(self.fitness, memo)
        return result

    def __lt__(self, other):
        if not self.fitness.valid or not other.fitness.valid:
            return False
        return self.fitness.values < other.fitness.values
        
    @property
    def unwrap(self):
        """Accedi alla strategia sottostante"""
        return self.strategy

class GeneticOptimizer:
    def __init__(self, strategy_class, config_loader, **kwargs):
        self.strategy_class = strategy_class
        self.config_loader = config_loader
        self.data_loader = DataLoader(config_loader)
        self.population_size = kwargs.get('population_size', 50)
        self.generations = kwargs.get('generations', 100)
        self.crossover_prob = kwargs.get('crossover_prob', 0.7)
        self.mutation_prob = kwargs.get('mutation_prob', 0.2)
        self.tournament_size = kwargs.get('tournament_size', 3)
        self.train_data = None
        self.test_data = None
        
        clear_deap()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", Individual)
        
        self.toolbox = base.Toolbox()
        self._setup_toolbox()
        
    def _create_individual(self):
        """Wrapper method for testing"""
        return self.toolbox.individual()
        
    def _mutate(self, individual):
        """Wrapper method for testing"""
        return self.toolbox.mutate(individual)
        
    def _crossover(self, ind1, ind2):
        """Wrapper method for testing"""
        return self.toolbox.mate(ind1, ind2)

    def _setup_toolbox(self):
        def create_individual(n=None):
            strategy = self.strategy_class(self.config_loader)
            individual = creator.Individual(strategy)
            print(f"Created individual with strategy: {strategy}")
            return individual
            
        def evaluate(individual):
            try:
                if not hasattr(individual, 'strategy') or individual.strategy is None:
                    print(f"Invalid individual without strategy: {individual}")
                    return (0.0,)
                
                if self.train_data is None:
                    print("No training data available for evaluation")
                    return (0.0,)
                    
                print(f"\nEvaluating individual: {individual}")
                print(f"Strategy: {individual.strategy}")
                print(f"Training data shape: {self.train_data.shape}")
                
                metrics = individual.strategy.evaluate(self.train_data)
                print(f"Raw evaluation metrics: {metrics}")
                
                if not isinstance(metrics, dict):
                    print(f"Invalid metrics type: {type(metrics)}")
                    return (0.0,)
                
                sharpe = float(metrics.get('sharpe_ratio', 0.0))
                dd = abs(float(metrics.get('max_drawdown', 0.0)))
                ret = max(0, float(metrics.get('annual_return', 0.0)))
                
                print(f"Processed metrics - Sharpe: {sharpe}, Drawdown: {dd}, Return: {ret}")
                
                fitness = sharpe + ret - dd
                print(f"Final fitness: {fitness}")
                
                return (fitness,)
            except Exception as e:
                logger.error(f"Evaluation error: {str(e)}")
                print(f"Evaluation failed with error: {str(e)}")
                return (0.0,)
                
        def mutate(individual):
            try:
                print(f"\nMutating individual: {individual}")
                individual.strategy.mutate()
                print("Mutation completed")
                return (individual,)
            except Exception as e:
                logger.error(f"Mutation error: {str(e)}")
                print(f"Mutation failed with error: {str(e)}")
                return (individual,)
            
        def mate(ind1, ind2):
            try:
                print(f"\nCrossover between: {ind1} and {ind2}")
                s1, s2 = ind1.strategy.crossover(ind2.strategy)
                child1 = creator.Individual(s1)
                child2 = creator.Individual(s2)
                print("Crossover completed")
                return (child1, child2)
            except Exception as e:
                logger.error(f"Crossover error: {str(e)}")
                print(f"Crossover failed with error: {str(e)}")
                return (ind1, ind2)
            
        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", mate)
        self.toolbox.register("mutate", mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def load_data(self, file_path, start_date=None, end_date=None, train_ratio=0.8):
        try:
            file_path = Path(file_path)
            if file_path.suffix == '.parquet':
                data = pd.read_parquet(file_path)
            else:
                data = pd.read_csv(file_path)
                
            # Converti timestamp
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
                
            # Filtra date
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
                
            # Split train/test
            split_idx = int(len(data) * train_ratio)
            self.train_data = data.iloc[:split_idx].copy()
            self.test_data = data.iloc[split_idx:].copy()
            
            if len(self.train_data) == 0 or len(self.test_data) == 0:
                raise ValueError("Training o test set vuoti dopo lo split")
                
        except Exception as e:
            logger.error(f"Errore caricamento dati: {e}")
            raise

    def optimize(self, callback=None):
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("No data loaded or insufficient data for optimization")
            
        # Popolazione iniziale
        pop = self.toolbox.population(n=self.population_size)
        
        # Assicurati che tutti gli individui abbiano un valore di fitness iniziale
        for ind in pop:
            if not hasattr(ind.fitness, 'values') or not ind.fitness.values:
                ind.fitness.values = (0.0,)
        
        # Setup statistiche
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = []
        
        # Valutazione iniziale popolazione
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        # Record statistiche iniziali
        record = stats.compile(pop)
        stats_dict = {
            'generation': 0,
            'avg': float(record['avg']),
            'std': float(record['std']),
            'min': float(record['min']),
            'max': float(record['max'])
        }
        logbook.append(stats_dict)
        
        if callback:
            callback(0, stats_dict)
            
        # Evoluzione
        for gen in range(1, self.generations + 1):
            # Selezione
            offspring = tools.selTournament(pop, len(pop), tournsize=self.tournament_size)
            offspring = list(map(copy.deepcopy, offspring))
            
            # Crossover
            for i in range(0, len(offspring), 2):
                if i+1 < len(offspring) and random.random() < self.crossover_prob:
                    child1, child2 = self.toolbox.mate(offspring[i], offspring[i+1])
                    offspring[i] = child1
                    offspring[i+1] = child2
                    offspring[i].fitness.values = (0.0,)  # Reset fitness con valore di default
                    offspring[i+1].fitness.values = (0.0,)  # Reset fitness con valore di default
            
            # Mutazione
            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    mutant = self.toolbox.mutate(offspring[i])[0]
                    offspring[i] = mutant
                    offspring[i].fitness.values = (0.0,)  # Reset fitness con valore di default
            
            # Valutazione
            invalid_ind = [ind for ind in offspring if not hasattr(ind.fitness, 'values')]
            for ind in invalid_ind:
                fitness = self.toolbox.evaluate(ind)
                if fitness is not None:
                    ind.fitness.values = fitness
                else:
                    ind.fitness.values = (0.0,)
            
            # Aggiorna popolazione
            pop[:] = offspring
            
            # Statistiche
            record = stats.compile(pop)
            stats_dict = {
                'generation': gen,
                'avg': float(record['avg']),
                'std': float(record['std']),
                'min': float(record['min']),
                'max': float(record['max'])
            }
            logbook.append(stats_dict)
            
            if callback:
                callback(gen, stats_dict)
        
        # Seleziona miglior individuo
        best = tools.selBest(pop, 1)[0]
        best_strategy = best.strategy
        best_strategy.fitness = float(best.fitness.values[0])
        
        return best_strategy, logbook

    def validate(self, strategy):
        if self.test_data is None or len(self.test_data) == 0:
            raise ValueError("Test data non disponibili o insufficienti")
        return strategy.evaluate(self.test_data)
