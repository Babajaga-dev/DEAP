import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import pandas as pd
from pathlib import Path
from src.download_binance import BinanceDataDownloader, parse_arguments
from src.data.data_loader import DataLoader
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def mock_binance_response():
    """Fixture che fornisce dati di esempio da Binance"""
    def create_kline(timestamp, open_price, high, low, close, volume):
        return [
            timestamp * 1000,  # Open time
            open_price,
            high,
            low,
            close,
            volume,
            timestamp * 1000 + 59999,  # Close time
            100.0,  # Quote asset volume
            10,     # Number of trades
            50.0,   # Taker buy base asset volume
            50.0,   # Taker buy quote asset volume
            0      # Ignore
        ]

    # Crea 3 candele di esempio
    now = int(datetime.now().timestamp())
    return [
        create_kline(now - 120, 100, 105, 95, 103, 1000),
        create_kline(now - 60, 103, 108, 102, 107, 1500),
        create_kline(now, 107, 110, 105, 108, 2000)
    ]

@pytest.fixture
def mock_config():
    """Fixture che fornisce una configurazione semplificata per i test"""
    return {
        "data": {
            "market": {
                "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "price_decimals": 8,
                "volume_decimals": 2
            },
            "preprocessing": {
                "handle_missing": {
                    "method": "drop",
                    "max_consecutive_missing": 5
                },
                "handle_outliers": {
                    "method": "none",
                    "threshold": 3.0
                },
                "volume_filter": {
                    "min_volume": 0,
                    "min_trades": 0
                }
            },
            "validation": {
                "price_checks": {
                    "high_low": True,
                    "open_close_range": True,
                    "zero_prices": False
                },
                "volume_checks": {
                    "min_volume": 0,
                    "max_volume": None
                },
                "timestamp_checks": {
                    "gaps": "ignore",
                    "duplicates": "ignore",
                    "timezone": "UTC"
                }
            }
        }
    }

@pytest.fixture
def config_loader(mock_config):
    """Fixture che fornisce un ConfigLoader configurato per i test"""
    loader = ConfigLoader()
    loader.load_config = Mock(return_value=mock_config)
    return loader

@pytest.fixture
def data_loader(config_loader):
    """Fixture che fornisce un DataLoader configurato per i test"""
    return DataLoader(config_loader)

def test_parse_arguments_defaults(monkeypatch):
    """Testa il parsing degli argomenti con valori di default"""
    monkeypatch.setattr("sys.argv", ["script"])
    args = parse_arguments()
    assert args.symbol is None
    assert args.interval is None
    assert args.start_date is None
    assert args.end_date is None
    assert args.output_folder is None

def test_parse_arguments_with_values(monkeypatch):
    """Testa il parsing degli argomenti con valori specificati"""
    test_args = [
        "script",
        "--symbol", "ETHUSDT",
        "--interval", "5m",
        "--start-date", "2024-03-01",
        "--output-folder", "custom/folder"
    ]
    monkeypatch.setattr("sys.argv", test_args)
    
    args = parse_arguments()
    assert args.symbol == "ETHUSDT"
    assert args.interval == "5m"
    assert args.start_date == "2024-03-01"
    assert args.output_folder == "custom/folder"
    assert args.end_date is None

@patch('src.download_binance.Client')
def test_download_historical_data(mock_client, mock_binance_response, tmp_path, data_loader, config_loader):
    """Testa il download e salvataggio dei dati storici"""
    # Configura il mock del client Binance
    mock_client_instance = Mock()
    mock_client_instance.get_historical_klines.return_value = mock_binance_response
    mock_client.return_value = mock_client_instance
    
    # Inizializza il downloader
    downloader = BinanceDataDownloader(config_loader)
    
    # Esegui il download
    output_file = downloader.download_historical_data(
        symbol="BTCUSDT",
        interval="1m",
        start_date="2024-01-01",
        end_date="2024-01-02",
        output_folder=str(tmp_path)
    )
    
    # Verifica che il file sia stato creato
    assert Path(output_file).exists()
    
    # Carica e verifica i dati usando DataLoader
    df = data_loader.load_csv(output_file)
    assert len(df) > 0, "Il DataFrame non dovrebbe essere vuoto"
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert df['open'].dtype == float
    assert df['volume'].dtype == float
    
    # Verifica le relazioni tra i prezzi
    assert (df['high'] >= df['low']).all(), "High deve essere >= Low"
    assert (df['high'] >= df['open']).all(), "High deve essere >= Open"
    assert (df['high'] >= df['close']).all(), "High deve essere >= Close"
    assert (df['low'] <= df['open']).all(), "Low deve essere <= Open"
    assert (df['low'] <= df['close']).all(), "Low deve essere <= Close"

@patch('src.download_binance.Client')
def test_error_handling(mock_client, config_loader):
    """Testa la gestione degli errori durante il download"""
    # Configura il mock per sollevare un'eccezione
    mock_client_instance = Mock()
    mock_client_instance.get_historical_klines.side_effect = Exception("API Error")
    mock_client.return_value = mock_client_instance
    
    downloader = BinanceDataDownloader(config_loader)
    
    with pytest.raises(Exception) as exc_info:
        downloader.download_historical_data(
            symbol="BTCUSDT",
            interval="1m",
            start_date="2024-01-01"
        )
    assert "API Error" in str(exc_info.value)

@patch('src.download_binance.Client')
def test_empty_response_handling(mock_client, tmp_path, config_loader):
    """Testa la gestione di risposte vuote da Binance"""
    # Configura il mock per restituire una lista vuota
    mock_client_instance = Mock()
    mock_client_instance.get_historical_klines.return_value = []
    mock_client.return_value = mock_client_instance
    
    downloader = BinanceDataDownloader(config_loader)
    
    with pytest.raises(Exception) as exc_info:
        downloader.download_historical_data(
            symbol="BTCUSDT",
            interval="1m",
            start_date="2024-01-01",
            output_folder=str(tmp_path)
        )
    assert "No data downloaded!" in str(exc_info.value)

@patch('src.download_binance.Client')
def test_invalid_interval(mock_client, config_loader):
    """Testa la gestione di intervalli temporali non validi"""
    downloader = BinanceDataDownloader(config_loader)
    
    with pytest.raises(KeyError):
        downloader.download_historical_data(
            symbol="BTCUSDT",
            interval="invalid",
            start_date="2024-01-01"
        )

def test_data_validation(mock_binance_response, tmp_path, data_loader):
    """Testa la validazione dei dati scaricati"""
    # Crea un DataFrame di test
    df = pd.DataFrame(mock_binance_response, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Salva il file
    output_file = tmp_path / "market_data_BTC_1m.csv"
    df.to_csv(output_file, index=False)
    
    # Carica e valida i dati usando DataLoader
    loaded_df = data_loader.load_csv(output_file)
    
    # Verifica la struttura dei dati
    assert isinstance(loaded_df.index, pd.DatetimeIndex)
    assert all(col in loaded_df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Verifica le relazioni tra i prezzi
    assert (loaded_df['high'] >= loaded_df['low']).all()
    assert (loaded_df['high'] >= loaded_df['open']).all()
    assert (loaded_df['high'] >= loaded_df['close']).all()
    assert (loaded_df['low'] <= loaded_df['open']).all()
    assert (loaded_df['low'] <= loaded_df['close']).all()
