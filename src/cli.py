#!/usr/bin/env python3
import click
import logging
import yaml
import os
from pathlib import Path
from typing import Optional
from .optimization.genetic_optimizer import GeneticOptimizer
from .strategies.trend_momentum_strategy import TrendMomentumStrategy
from .strategies.options_strategy import OptionsStrategy
from .utils.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRATEGY_MAPPING = {
    'trend': TrendMomentumStrategy,
    'options': OptionsStrategy
}

STRATEGY_CONFIG_NAMES = {
    'trend': 'trendmomentum',
    'options': 'options'
}

def get_config_dir() -> Path:
    """Get configuration directory from environment or default"""
    config_dir = os.getenv('CONFIG_DIR')
    if config_dir:
        return Path(config_dir)
    return Path('config')

def print_help_examples():
    """Stampa esempi di utilizzo del CLI"""
    click.echo("""
Esempi di utilizzo:

1. Ottimizzazione di una strategia:
   python -m src.cli optimize --strategy trend --data data/test_data.parquet --generations 50

2. Backtest di una strategia salvata:
   python -m src.cli backtest --strategy trend --data data/test_data.parquet --model saved_strategy.json

3. Visualizzazione configurazione:
   python -m src.cli config show --type genetic

4. Modifica configurazione:
   python -m src.cli config edit --type genetic --param population_size --value 100
""")

@click.group()
@click.option('--debug/--no-debug', default=False, help='Abilita modalità debug')
def cli(debug):
    """CLI per ottimizzazione genetica di strategie di trading"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option('--strategy', type=click.Choice(['trend', 'options']), required=True,
              help='Tipo di strategia da ottimizzare')
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Path del file dati (CSV o Parquet)')
@click.option('--generations', type=int, default=100,
              help='Numero di generazioni per ottimizzazione')
@click.option('--population', type=int, default=50,
              help='Dimensione della popolazione')
@click.option('--output', type=click.Path(), default='best_strategy.json',
              help='Path dove salvare la strategia ottimizzata')
def optimize(strategy: str, data: str, generations: int, population: int, output: str):
    """Ottimizza una strategia usando algoritmi genetici"""
    try:
        config_loader = ConfigLoader(config_dir=get_config_dir())
        strategy_class = STRATEGY_MAPPING[strategy]
        strategy_config_name = STRATEGY_CONFIG_NAMES[strategy]
        
        # Carica configurazione strategia
        try:
            config_loader.get_strategy_config(strategy_config_name)
        except ValueError as e:
            raise click.ClickException(f"Strategia {strategy_config_name} non trovata nella configurazione")
            
        optimizer = GeneticOptimizer(
            strategy_class=strategy_class,
            config_loader=config_loader,
            population_size=population,
            generations=generations
        )
        
        # Carica dati
        logger.info(f"Caricamento dati da {data}")
        optimizer.load_data(data)
        
        # Callback per logging
        def log_generation(gen: int, stats: dict):
            logger.info(f"Generazione {gen}: max={stats['max']:.4f}, avg={stats['avg']:.4f}")
        
        # Esegui ottimizzazione
        logger.info("Avvio ottimizzazione...")
        best_strategy, logbook = optimizer.optimize(callback=log_generation)
        
        # Valida su test set
        logger.info("Validazione su test set...")
        test_metrics = optimizer.validate(best_strategy)
        
        # Salva risultati
        strategy_dict = best_strategy.to_dict()
        strategy_dict['test_metrics'] = test_metrics
        
        with open(output, 'w') as f:
            yaml.dump(strategy_dict, f)
            
        logger.info(f"Strategia ottimizzata salvata in {output}")
        logger.info(f"Metriche test: {test_metrics}")
        
    except Exception as e:
        logger.error(f"Errore durante ottimizzazione: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--strategy', type=click.Choice(['trend', 'options']), required=True,
              help='Tipo di strategia da testare')
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Path del file dati (CSV o Parquet)')
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Path del file strategia salvata')
def backtest(strategy: str, data: str, model: str):
    """Esegui backtest di una strategia salvata"""
    try:
        config_loader = ConfigLoader(config_dir=get_config_dir())
        strategy_class = STRATEGY_MAPPING[strategy]
        strategy_config_name = STRATEGY_CONFIG_NAMES[strategy]
        
        # Carica configurazione strategia
        try:
            config_loader.get_strategy_config(strategy_config_name)
        except ValueError as e:
            raise click.ClickException(f"Strategia {strategy_config_name} non trovata nella configurazione")
        
        # Carica strategia
        with open(model, 'r') as f:
            strategy_dict = yaml.safe_load(f)
            
        strategy = strategy_class(config_loader)
        strategy.from_dict(strategy_dict)
        
        # Carica dati
        optimizer = GeneticOptimizer(strategy_class, config_loader)
        optimizer.load_data(data)
        
        # Esegui backtest
        metrics = optimizer.validate(strategy)
        logger.info(f"Risultati backtest: {metrics}")
        
    except Exception as e:
        logger.error(f"Errore durante backtest: {str(e)}")
        raise click.ClickException(str(e))

@cli.group()
def config():
    """Gestione configurazioni"""
    pass

@config.command()
@click.option('--type', type=click.Choice(['genetic', 'backtest', 'data', 'indicators', 'strategies']),
              required=True, help='Tipo di configurazione')
def show(type: str):
    """Mostra configurazione corrente"""
    try:
        config_loader = ConfigLoader(config_dir=get_config_dir())
        if type == 'genetic':
            config = config_loader.get_genetic_config()
        elif type == 'backtest':
            config = config_loader.get_backtest_config()
        elif type == 'data':
            config = config_loader.get_data_config()
        elif type == 'indicators':
            config = config_loader.get_indicators_config()
        elif type == 'strategies':
            config = config_loader.get_strategy_config()
            
        click.echo(yaml.dump(config))
        
    except Exception as e:
        logger.error(f"Errore lettura configurazione: {str(e)}")
        raise click.ClickException(str(e))

@config.command()
@click.option('--type', type=click.Choice(['genetic', 'backtest', 'data', 'indicators', 'strategies']),
              required=True, help='Tipo di configurazione')
@click.option('--param', required=True, help='Nome parametro da modificare')
@click.option('--value', required=True, help='Nuovo valore')
def edit(type: str, param: str, value: str):
    """Modifica parametro di configurazione"""
    try:
        config_dir = get_config_dir()
        config_path = config_dir / f'{type}.yaml'
        if not config_path.exists():
            raise click.ClickException(f"File configurazione {type}.yaml non trovato")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Converti value al tipo corretto
        try:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
        except:
            pass  # Mantieni come stringa
            
        # Modifica configurazione
        config_keys = param.split('.')
        current = config
        for key in config_keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[config_keys[-1]] = value
        
        # Salva configurazione
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        logger.info(f"Configurazione {type} aggiornata")
        
    except Exception as e:
        logger.error(f"Errore modifica configurazione: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
def examples():
    """Mostra esempi di utilizzo"""
    print_help_examples()

if __name__ == '__main__':
    cli()
