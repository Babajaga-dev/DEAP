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
│   ├── indicators.yaml          # Technical indicators parameters
│   ├── strategies.yaml          # Strategy parameters
│   ├── genetic.yaml             # Genetic optimization parameters
│   ├── backtest.yaml            # Backtesting parameters
│   └── data.yaml               # Data handling parameters
├── data/
│   ├── input/                  # Raw data files (CSV, Parquet)
│   ├── output/                 # Processed data files
│   └── cache/                  # Temporary data files
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

3. Install Binance Connector:
```bash
pip install binance-connector
```

4. Install TA-Lib:
   
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

### Data Setup

1. Place your raw data files in the `data/input` directory
2. The system supports CSV and Parquet formats with the following columns:
```
timestamp,open,high,low,close,volume
2024-11-14 23:00:00,9.93e-05,9.959e-05,9.93e-05,9.956e-05,91042100.1
2024-11-14 23:01:00,9.945e-05,9.945e-05,9.916e-05,9.916e-05,111558380.04
...
```

Data directories can be configured in `config/data.yaml`:
```yaml
data:
  directories:
    input: "data/input"    # Directory for raw data
    output: "data/output"  # Directory for processed data
    cache: "data/cache"    # Directory for temporary files
```

### Download Data from Binance

Download historical market data from Binance using the download command:
```bash
python -m src.download_binance
```

The system uses the official Binance Connector library for reliable and efficient data downloading. Parameters can be configured in `config/data.yaml` under the `data.download` section:
```yaml
data:
  download:
    symbol: "BTCUSDT"          # Trading pair
    interval: "1m"             # Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    start_date: "2024-01-01"   # Start date
    end_date: null             # End date (null for current date)
    batch_size: 1000           # Number of candles per request
    output_folder: "data/input" # Output directory
```

You can also override these parameters from command line:
```bash
# Download ETHUSDT 5-minute candles for February 2024
python -m src.download_binance --symbol ETHUSDT --interval 5m --start-date 2024-02-01 --end-date 2024-02-29

# Download BTCUSDT hourly data for the last 30 days
python -m src.download_binance --symbol BTCUSDT --interval 1h

# Download current month's data with custom output folder
python -m src.download_binance --symbol BTCUSDT --start-date 2024-03-01 --output-folder data/custom_folder
```

Available parameters:
- `--symbol`: Trading pair (e.g., BTCUSDT, ETHUSDT)
- `--interval`: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format (optional)
- `--output-folder`: Output directory

Note: 
- API keys are not required for downloading historical data. While they can be configured to increase rate limits, they are completely optional.
- The system uses the official Binance Connector library which provides stable and reliable access to Binance's API with proper rate limiting and error handling.

### Strategy Optimization

Optimize a trading strategy using genetic algorithms:
```bash
python -m src.cli optimize --strategy trend --data data/input/test_data.parquet --generations 50
```

Available strategies:
- `trend`: Trend following strategy using RSI, MACD, and Bollinger Bands
- `options`: Options strategy using RSI, Bollinger Bands, and ATR

### Strategy Backtest

Run backtest on a saved strategy:
```bash
python -m src.cli backtest --strategy trend --data data/input/test_data.parquet --model saved_strategy.json
```

### Configuration Management

View current configuration:
```bash
python -m src.cli config show --type genetic
```

Modify configuration:
```bash
python -m src.cli config edit --type genetic --param optimization.population_size --value 100
```

Available configuration types:
- `genetic`: Genetic optimization parameters
- `backtest`: Backtesting parameters
- `data`: Data handling parameters
- `indicators`: Technical indicators parameters
- `strategies`: Strategy parameters

### Help and Examples

Show usage examples:
```bash
python -m src.cli examples
```

Show help for any command:
```bash
python -m src.cli --help
python -m src.cli optimize --help
python -m src.cli config --help
```

## Configuration

Each component can be configured through YAML files in the `config` directory:

- `data.yaml`: Data processing parameters and directory paths
- `indicators.yaml`: Technical indicator parameters
- `strategies.yaml`: Strategy-specific parameters
- `genetic.yaml`: Genetic optimization parameters
- `backtest.yaml`: Backtesting parameters

## Testing

Run all tests:
```bash
python -m pytest tests/
```

Run specific test suite:
```bash
python -m pytest tests/data/test_data_loader.py
python -m pytest tests/strategies/test_options_strategy.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -am 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a new Pull Request
