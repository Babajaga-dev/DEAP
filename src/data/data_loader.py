import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path
import logging
from ..utils.config_loader import ConfigLoader

class DataLoader:
    """Data loader with configuration support"""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """Initialize DataLoader with configuration"""
        if config_loader is None:
            config_loader = ConfigLoader()
            
        self.config_loader = config_loader
        self.config = self.config_loader.load_config("data")["data"]
        self._data = None
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load historical data from CSV file
        
        Format expected from config['market']['required_columns']
        """
        data = pd.read_csv(file_path)
        
        # Validate columns
        required_columns = self.config["market"]["required_columns"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Check for negative volumes
        if (data['volume'] < 0).any():
            raise ValueError("Volume below minimum: negative volume not allowed")
        
        # Handle missing values first
        if pd.isnull(data).any().any():
            method = self.config["preprocessing"]["handle_missing"]["method"]
            max_missing = self.config["preprocessing"]["handle_missing"]["max_consecutive_missing"]
            
            if method == "forward_fill":
                data = data.ffill(limit=max_missing)
            elif method == "backfill":
                data = data.bfill(limit=max_missing)
            elif method == "interpolate":
                data = data.interpolate(limit=max_missing)
            elif method == "drop":
                data = data.dropna()
            
            # Dopo il fill, aggiusta i prezzi per mantenere le relazioni
            for idx in data.index:
                # Assicura che high sia il massimo
                data.loc[idx, 'high'] = max(
                    data.loc[idx, 'open'],
                    data.loc[idx, 'close'],
                    data.loc[idx, 'high']
                )
                # Assicura che low sia il minimo
                data.loc[idx, 'low'] = min(
                    data.loc[idx, 'open'],
                    data.loc[idx, 'close'],
                    data.loc[idx, 'low']
                )
                
        # Verify price relationships after handling missing values
        if self.config["validation"]["price_checks"]["high_low"]:
            if not (data['high'] >= data['low']).all():
                raise ValueError("High price must be >= Low price")
                
        if self.config["validation"]["price_checks"]["open_close_range"]:
            if not ((data['high'] >= data['open']) & 
                   (data['high'] >= data['close']) &
                   (data['low'] <= data['open']) & 
                   (data['low'] <= data['close'])).all():
                raise ValueError("Open/Close must be within High/Low range")
        
        # Handle outliers
        if self.config["preprocessing"]["handle_outliers"]["method"] == "zscore":
            threshold = self.config["preprocessing"]["handle_outliers"]["threshold"]
            # Calcola gli z-scores per ogni colonna di prezzo separatamente
            for col in ['open', 'high', 'low', 'close']:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < threshold]
        
        # Apply volume filter
        min_volume = self.config["preprocessing"]["volume_filter"]["min_volume"]
        if min_volume:
            data = data[data['volume'] >= min_volume]
        
        # Round values
        decimals = self.config["market"]["price_decimals"]
        volume_decimals = self.config["market"]["volume_decimals"]
        
        for col in ['open', 'high', 'low', 'close']:
            data[col] = data[col].round(decimals)
        data['volume'] = data['volume'].round(volume_decimals)
        
        # Verify remaining data integrity rules after preprocessing
        if not self.config["validation"]["price_checks"]["zero_prices"]:
            if (data[['open', 'high', 'low', 'close']] == 0).any().any():
                raise ValueError("Zero prices found but not allowed")
                
        # Volume checks after preprocessing
        min_volume = self.config["preprocessing"]["volume_filter"]["min_volume"]
        if min_volume and (data['volume'] < min_volume).any():
            raise ValueError(f"Volume below minimum: {min_volume}")
            
        max_volume = self.config["validation"]["volume_checks"]["max_volume"]
        if max_volume is not None and (data['volume'] > max_volume).any():
            raise ValueError(f"Volume above maximum: {max_volume}")
        
        # Timestamp checks
        if self.config["validation"]["timestamp_checks"]["duplicates"] == "error":
            if data.index.duplicated().any():
                raise ValueError("Duplicate timestamps found")
        
        self._data = data
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps from configuration"""
        config = self.config["preprocessing"]
        
        # Handle missing values
        if pd.isnull(data).any().any():
            method = config["handle_missing"]["method"]
            max_missing = config["handle_missing"]["max_consecutive_missing"]
            
            if method == "forward_fill":
                data = data.fillna(method='ffill', limit=max_missing)
            elif method == "backfill":
                data = data.fillna(method='bfill', limit=max_missing)
            elif method == "interpolate":
                data = data.interpolate(limit=max_missing)
            elif method == "drop":
                data = data.dropna()
        
        # Handle outliers
        if config["handle_outliers"]["method"] == "zscore":
            threshold = config["handle_outliers"]["threshold"]
            z_scores = np.abs((data - data.mean()) / data.std())
            data = data[z_scores < threshold]
        
        # Apply volume filter
        min_volume = config["volume_filter"]["min_volume"]
        if min_volume:
            data = data[data['volume'] >= min_volume]
        
        # Round values
        decimals = self.config["market"]["price_decimals"]
        volume_decimals = self.config["market"]["volume_decimals"]
        
        for col in ['open', 'high', 'low', 'close']:
            data[col] = data[col].round(decimals)
        data['volume'] = data['volume'].round(volume_decimals)
        
        return data
    
    def _verify_data(self, data: pd.DataFrame) -> None:
        """Verify data integrity based on configuration"""
        validation = self.config["validation"]
        
        # Price checks
        if validation["price_checks"]["high_low"]:
            if not (data['high'] >= data['low']).all():
                raise ValueError("High price must be >= Low price")
        
        if validation["price_checks"]["open_close_range"]:
            if not ((data['high'] >= data['open']) & 
                   (data['high'] >= data['close']) &
                   (data['low'] <= data['open']) & 
                   (data['low'] <= data['close'])).all():
                raise ValueError("Open/Close must be within High/Low range")
        
        if not validation["price_checks"]["zero_prices"]:
            if (data[['open', 'high', 'low', 'close']] == 0).any().any():
                raise ValueError("Zero prices found but not allowed")
        
        # Volume checks
        min_volume = self.config["preprocessing"]["volume_filter"]["min_volume"]
        if min_volume and (data['volume'] < min_volume).any():
            raise ValueError(f"Volume below minimum: {min_volume}")
            
        max_volume = validation["volume_checks"]["max_volume"]
        if max_volume is not None and (data['volume'] > max_volume).any():
            raise ValueError(f"Volume above maximum: {max_volume}")
        
        # Timestamp checks
        if validation["timestamp_checks"]["duplicates"] == "error":
            if data.index.duplicated().any():
                raise ValueError("Duplicate timestamps found")
    
    def get_timeframe_data(self, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Get data for specific timeframe"""
        if self._data is None:
            raise ValueError("No data loaded. Call load_csv first.")
            
        data = self._data.copy()
        
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]
            
        return data
    
    def get_train_test_split(self, 
                            train_ratio: Optional[float] = None,
                            validation_ratio: Optional[float] = None,
                            shuffle: Optional[bool] = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], 
                                                                   Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Split data into training and test (and validation) sets"""
        if self._data is None:
            raise ValueError("No data loaded. Call load_csv first.")
            
        # Use configuration if parameters not provided
        split_config = self.config["split"]
        train_ratio = train_ratio if train_ratio is not None else split_config["train_ratio"]
        # Se validation_ratio è None, non usare il valore dalla configurazione
        validation_ratio = validation_ratio
        shuffle = shuffle if shuffle is not None else split_config["shuffle"]
        
        if shuffle:
            data = self._data.sample(frac=1, random_state=42)
        else:
            data = self._data.copy()
            
        n = len(data)
        train_idx = int(n * train_ratio)
        
        if validation_ratio:
            val_idx = int(n * (train_ratio + validation_ratio))
            train_data = data.iloc[:train_idx]
            val_data = data.iloc[train_idx:val_idx]
            test_data = data.iloc[val_idx:]
            return train_data, val_data, test_data
        else:
            train_data = data.iloc[:train_idx]
            test_data = data.iloc[train_idx:]
            return train_data, test_data
            
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save data according to configuration"""
        save_config = self.config["save"]
        save_dir = Path(save_config["directory"])
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / filename
        format = save_config["format"]
        
        if format == "parquet":
            data.to_parquet(filepath, compression=save_config["compression"])
        elif format == "csv":
            data.to_csv(filepath)
        elif format == "pickle":
            data.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        self.logger.info(f"Data saved to {filepath}")
