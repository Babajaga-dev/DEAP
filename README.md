Creare ambiente
py -m venv DEAP

Attivare
.\DEAP\Scripts\Activate

# Trading System Based on DNA and Genes

This project implements a trading system using genetic algorithms and evolutionary strategies. The system uses concepts inspired by DNA and genes to optimize trading strategies within a simulation environment and, subsequently, in real trading contexts.

## Project Structure

```
genetic-trading/
├── config/
│   ├── config.yaml              # Main configuration
│   ├── indicators.yaml          # Technical indicators parameters
│   ├── strategies.yaml          # Strategy parameters
│   ├── genetic.yaml             # Genetic optimization parameters
│   ├── backtest.yaml            # Backtesting parameters
│   └── data.yaml               # Data handling parameters
├── src/
│   ├── genes/                  # Gene implementations
│   ├── strategies/             # Trading strategies
│   ├── optimization/           # Genetic optimization
│   ├── data/                   # Data handling
│   └── utils/                  # Utilities
├── tests/                      # Unit tests
└── cli.py                      # Command Line Interface
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genetic-trading.git
cd genetic-trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install TA-Lib:
   
For Linux:
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

For Windows:
- Download the binary from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- Install with: `pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl`

## Usage

### Data Preparation

Process and validate historical data:
```bash
python cli.py prepare-data \
  --data-path data/raw/historical_data.csv \
  --output-dir data/processed \
  --train-ratio 0.8
```

### Strategy Optimization

Optimize a trading strategy using genetic algorithms:
```bash
python cli.py optimize \
  --strategy options \
  --data-path data/processed/train_data.parquet \
  --output-dir results
```

Available strategies:
- `trend_momentum`: Trend following strategy using RSI, MACD, and Bollinger Bands
- `options`: Options strategy using RSI, Bollinger Bands, and ATR

### Strategy Validation

Validate an optimized strategy on test data:
```bash
python cli.py validate \
  --strategy-path results/options_results_20240122_120000.yaml \
  --data-path data/processed/test_data.parquet \
  --output-dir validation
```

### Performance Report

Generate a performance report from validation results:
```bash
python cli.py report \
  --results-path validation/validation_results_20240122_120000.yaml
```

## Data Format

The system expects historical data in CSV format with the following columns:
```
timestamp,open,high,low,close,volume
2024-11-14 23:00:00,9.93e-05,9.959e-05,9.93e-05,9.956e-05,91042100.1
2024-11-14 23:01:00,9.945e-05,9.945e-05,9.916e-05,9.916e-05,111558380.04
...
```

## Configuration

Each component can be configured through YAML files in the `config` directory:

- `data.yaml`: Data processing parameters
- `indicators.yaml`: Technical indicator parameters
- `strategies.yaml`: Strategy-specific parameters
- `genetic.yaml`: Genetic optimization parameters
- `backtest.yaml`: Backtesting parameters

## Testing

Run unit tests:
```bash
pytest tests/
```

Run specific test suite:
```bash
pytest tests/data/test_data_loader.py
pytest tests/strategies/test_options_strategy.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -am 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a new Pull Request