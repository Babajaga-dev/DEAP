import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.optimization.genetic_optimizer import GeneticOptimizer
from src.strategies.base_strategy import BaseStrategy, StrategyConfig
from src.utils.config_loader import ConfigLoader

class MockStrategy(BaseStrategy):
    def _initialize_genes(self) -> None:
        """Initialize strategy genes"""
        self._genes = {}
        
    def generate_signals(self, data):
        return pd.Series(np.random.choice([-1, 0, 1], size=len(data)), index=data.index)
        
    def mutate(self):
        pass
        
    def crossover(self, other):
        child1 = MockStrategy(self.config_loader)
        child2 = MockStrategy(self.config_loader)
        return child1, child2

    def to_dict(self) -> dict:
        """Convert strategy to dictionary"""
        return {
            'name': 'Mock Strategy',
            'genes': {},
            'config': {
                'name': 'Mock Strategy',
                'version': '1.0.0',
                'description': None,
                'position_sizing': {'method': 'fixed', 'base_size': 1.0},
                'indicators': ['rsi', 'bollinger', 'atr']
            }
        }

    def from_dict(self, data: dict) -> None:
        """Initialize strategy from dictionary"""
        self._genes = {}  # Reset genes

@pytest.fixture
def market_data(tmp_path):
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    
    # Genera prima i prezzi low e high
    low_prices = np.random.uniform(8.5e-5, 9e-5, 100)
    high_prices = np.random.uniform(1.05e-4, 1.1e-4, 100)  # Sempre maggiore di low
    
    # Genera open e close tra low e high
    open_prices = np.array([np.random.uniform(low, high) for low, high in zip(low_prices, high_prices)])
    close_prices = np.array([np.random.uniform(low, high) for low, high in zip(low_prices, high_prices)])
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.uniform(90000000, 120000000, 100)
    })
    
    file_path = tmp_path / "test_data.csv"
    data.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def config_loader():
    return ConfigLoader()

@pytest.fixture
def optimizer(config_loader):
    return GeneticOptimizer(
        strategy_class=MockStrategy,
        config_loader=config_loader,
        population_size=10,
        generations=5
    )

class TestGeneticOptimizer:
    def test_initialization(self, optimizer):
        assert optimizer.population_size == 10
        assert optimizer.generations == 5
        assert optimizer.strategy_class == MockStrategy
        
    def test_load_data(self, optimizer, market_data):
        optimizer.load_data(market_data, train_ratio=0.8)
        assert optimizer.train_data is not None
        assert optimizer.test_data is not None
        assert len(optimizer.train_data) > len(optimizer.test_data)
        
    def test_optimization_without_data(self, optimizer):
        with pytest.raises(ValueError, match="No data loaded"):
            optimizer.optimize()
            
    def test_optimization_basic(self, optimizer, market_data):
        optimizer.load_data(market_data)
        best_strategy, logbook = optimizer.optimize()
        
        assert isinstance(best_strategy, MockStrategy)
        # Il logbook include la generazione iniziale (gen 0) più le generazioni evolutive
        assert len(logbook) == optimizer.generations + 1
        assert all(metric in logbook[0] for metric in ['avg', 'std', 'min', 'max'])
        
    def test_optimization_improvement(self, optimizer, market_data):
        optimizer.load_data(market_data)
        best_strategy, logbook = optimizer.optimize()
        
        # Verifica che ci sia variazione nel fitness tra le generazioni
        # Non possiamo garantire un miglioramento con segnali casuali
        fitness_values = [gen['max'] for gen in logbook]
        assert len(set(fitness_values)) > 1, "Il fitness dovrebbe variare tra le generazioni"
        
    def test_validation(self, optimizer, market_data):
        optimizer.load_data(market_data)
        best_strategy, _ = optimizer.optimize()
        
        validation_metrics = optimizer.validate(best_strategy)
        assert isinstance(validation_metrics, dict)
        assert all(metric in validation_metrics for metric in 
                  ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown'])
        
    def test_callback(self, optimizer, market_data):
        callback_called = []
        
        def callback(gen, stats):
            callback_called.append((gen, stats))
            
        optimizer.load_data(market_data)
        optimizer.optimize(callback=callback)
        
        assert len(callback_called) == optimizer.generations + 1  # Include la generazione iniziale
        assert all(isinstance(stats, dict) for _, stats in callback_called)
        
    def test_population_evolution(self, optimizer, market_data):
        optimizer.load_data(market_data)
        
        # Track population fitness over generations
        fitness_history = []
        def callback(gen, stats):
            fitness_history.append(stats['avg'])
            
        optimizer.optimize(callback=callback)
        
        # Check if population shows some variation
        fitness_std = np.std(fitness_history)
        assert fitness_std > 0
        
    def test_mutation_effect(self, optimizer, market_data):
        optimizer.mutation_prob = 1.0  # Force mutation
        optimizer.load_data(market_data)
        
        strategy = optimizer._create_individual()
        mutated = optimizer._mutate(strategy)[0]
        
        # In MockStrategy mutation doesn't change anything, 
        # but we test the interface is working
        assert isinstance(mutated.unwrap, MockStrategy)
        
    def test_crossover_effect(self, optimizer, market_data):
        optimizer.crossover_prob = 1.0  # Force crossover
        optimizer.load_data(market_data)
        
        parent1 = optimizer._create_individual()
        parent2 = optimizer._create_individual()
        child1, child2 = optimizer._crossover(parent1, parent2)
        
        assert isinstance(child1.unwrap, MockStrategy)
        assert isinstance(child2.unwrap, MockStrategy)
        
    def test_timeframe_optimization(self, optimizer, market_data):
        # Leggi i dati per determinare il range di date disponibili
        data = pd.read_csv(market_data)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        
        # Prendi solo i primi 3 giorni di dati
        end_date = data.index[0].normalize() + pd.Timedelta(days=2)
        filtered_data = data[data.index <= end_date]
        
        # Salva i dati filtrati in un nuovo file
        filtered_path = Path(market_data).parent / "filtered_data.csv"
        filtered_data.to_csv(filtered_path)
        
        # Usa i dati filtrati per il test
        optimizer.load_data(filtered_path)
        
        assert optimizer.train_data.index[0].normalize() >= data.index[0].normalize()
        assert optimizer.train_data.index[-1].normalize() <= end_date
