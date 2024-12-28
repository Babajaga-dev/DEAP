import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.data.data_loader import DataLoader
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def mock_data_config():
    """Create mock data configuration"""
    return {
        "data": {
            "market": {
                "timeframe": "1h",
                "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "price_decimals": 8,
                "volume_decimals": 2
            },
            "preprocessing": {
                "handle_missing": {
                    "method": "forward_fill",
                    "max_consecutive_missing": 5
                },
                "handle_outliers": {
                    "method": "zscore",
                    "threshold": 3.0
                },
                "volume_filter": {
                    "min_volume": 1000000,
                    "min_trades": 100
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
                    "gaps": "warn",
                    "duplicates": "error",
                    "timezone": "UTC"
                }
            },
            "split": {
                "train_ratio": 0.8,
                "validation_ratio": 0.1,
                "shuffle": False,
                "stratify": False
            },
            "save": {
                "format": "parquet",
                "compression": "snappy",
                "directory": "data"
            }
        }
    }

@pytest.fixture
def valid_data():
    """Create valid market data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(9e-5, 1e-4, 100),
        'high': np.random.uniform(9.5e-5, 1.1e-4, 100),
        'low': np.random.uniform(8.5e-5, 9e-5, 100),
        'close': np.random.uniform(9e-5, 1e-4, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    })
    
    # Ensure high/low relationships
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    return data

@pytest.fixture
def config_loader(tmp_path, mock_data_config):
    """Create ConfigLoader with mock configuration"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    import yaml
    with open(config_dir / "data.yaml", 'w') as f:
        yaml.dump(mock_data_config, f)
    
    loader = ConfigLoader()
    loader.config_dir = str(config_dir)
    return loader

@pytest.fixture
def data_loader(config_loader):
    """Create DataLoader instance"""
    return DataLoader(config_loader)

class TestDataLoader:
    def test_initialization(self, data_loader):
        """Test DataLoader initialization"""
        assert data_loader is not None
        assert data_loader.config is not None
        assert "market" in data_loader.config
        
    def test_load_csv_valid_data(self, data_loader, valid_data, tmp_path):
        """Test loading valid CSV data"""
        csv_path = tmp_path / "valid_data.csv"
        valid_data.to_csv(csv_path, index=False)
        
        loaded_data = data_loader.load_csv(csv_path)
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.index.name == 'timestamp'
        assert all(col in loaded_data.columns 
                  for col in ['open', 'high', 'low', 'close', 'volume'])
        
    def test_load_csv_missing_columns(self, data_loader, tmp_path):
        """Test loading CSV with missing columns"""
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10),
            'close': np.random.random(10)  # Missing required columns
        })
        
        csv_path = tmp_path / "invalid_data.csv"
        invalid_data.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="CSV must contain columns"):
            data_loader.load_csv(csv_path)
            
    def test_preprocessing(self, data_loader, valid_data, tmp_path):
        """Test data preprocessing"""
        # Add some missing values
        valid_data.loc[5:7, 'close'] = np.nan
        
        csv_path = tmp_path / "data_with_missing.csv"
        valid_data.to_csv(csv_path, index=False)
        
        processed_data = data_loader.load_csv(csv_path)
        assert not processed_data.isnull().any().any()  # Missing values handled
        
    def test_outlier_handling(self, data_loader, valid_data, tmp_path):
        """Test outlier handling"""
        # Add outlier while maintaining price relationships
        base_value = valid_data['close'].mean()
        outlier_value = base_value + 10 * valid_data['close'].std()

        # Aggiorniamo i valori mantenendo le relazioni corrette
        valid_data.loc[5, 'open'] = outlier_value
        valid_data.loc[5, 'close'] = outlier_value
        valid_data.loc[5, 'high'] = outlier_value * 1.1  # 10% sopra l'outlier
        valid_data.loc[5, 'low'] = outlier_value * 0.9   # 10% sotto l'outlier

        # Ensure high/low relationships for all rows
        for idx in valid_data.index:
            # Ensure high is the maximum
            valid_data.loc[idx, 'high'] = max(
                valid_data.loc[idx, 'open'],
                valid_data.loc[idx, 'close'],
                valid_data.loc[idx, 'high']
            )
            # Ensure low is the minimum
            valid_data.loc[idx, 'low'] = min(
                valid_data.loc[idx, 'open'],
                valid_data.loc[idx, 'close'],
                valid_data.loc[idx, 'low']
            )
        
        csv_path = tmp_path / "data_with_outliers.csv"
        valid_data.to_csv(csv_path, index=False)
        
        processed_data = data_loader.load_csv(csv_path)
        assert len(processed_data) < len(valid_data)  # Outlier removed
        
    def test_price_validation(self, data_loader, valid_data, tmp_path):
        """Test price validation"""
        # Create invalid price relationships
        valid_data.loc[5, 'high'] = valid_data.loc[5, 'low'] - 0.1
        
        csv_path = tmp_path / "invalid_prices.csv"
        valid_data.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="High price must be >= Low price"):
            data_loader.load_csv(csv_path)
            
    def test_volume_validation(self, data_loader, valid_data, tmp_path):
        """Test volume validation"""
        # Set invalid volume
        valid_data.loc[5, 'volume'] = -1000
        
        csv_path = tmp_path / "invalid_volume.csv"
        valid_data.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="negative volume not allowed"):
            data_loader.load_csv(csv_path)
            
    def test_timestamp_validation(self, data_loader, valid_data, tmp_path):
        """Test timestamp validation"""
        # Add duplicate timestamp
        valid_data.loc[1, 'timestamp'] = valid_data.loc[0, 'timestamp']
        
        csv_path = tmp_path / "duplicate_timestamps.csv"
        valid_data.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Duplicate timestamps found"):
            data_loader.load_csv(csv_path)
            
    def test_timeframe_filtering(self, data_loader, valid_data, tmp_path):
        """Test timeframe filtering"""
        csv_path = tmp_path / "full_data.csv"
        valid_data.to_csv(csv_path, index=False)
        
        data_loader.load_csv(csv_path)
        
        start_date = '2024-01-02'
        end_date = '2024-01-03'
        filtered_data = data_loader.get_timeframe_data(start_date, end_date)
        
        assert filtered_data.index.min().strftime('%Y-%m-%d') >= start_date
        assert filtered_data.index.max().strftime('%Y-%m-%d') <= end_date
        
    def test_train_test_split(self, data_loader, valid_data, tmp_path):
        """Test train/test split"""
        csv_path = tmp_path / "split_data.csv"
        valid_data.to_csv(csv_path, index=False)
        
        data_loader.load_csv(csv_path)
        
        # Test without validation set
        train_data, test_data = data_loader.get_train_test_split(validation_ratio=None)
        assert len(train_data) > len(test_data)
        assert len(train_data) + len(test_data) == len(valid_data)
        
        # Test with validation set
        train_data, val_data, test_data = data_loader.get_train_test_split(train_ratio=0.7, validation_ratio=0.2)
        assert len(train_data) > len(val_data) > len(test_data)
        assert len(train_data) + len(val_data) + len(test_data) == len(valid_data)
        
    def test_shuffled_split(self, data_loader, valid_data, tmp_path):
        """Test train/test split with shuffling"""
        csv_path = tmp_path / "shuffle_data.csv"
        valid_data.to_csv(csv_path, index=False)
        
        data_loader.load_csv(csv_path)
        
        train1, test1 = data_loader.get_train_test_split(shuffle=False)
        train2, test2 = data_loader.get_train_test_split(shuffle=True)
        
        assert not train1.index.equals(train2.index)
        
    def test_save_data(self, data_loader, valid_data, tmp_path):
        """Test data saving"""
        # Test different formats
        formats = ['parquet', 'csv', 'pickle']
        for fmt in formats:
            save_path = tmp_path / f"test_data.{fmt}"
            data_loader.save_data(valid_data, save_path)
            assert save_path.exists()
            
    def test_data_rounding(self, data_loader, valid_data, tmp_path):
        """Test price and volume rounding"""
        csv_path = tmp_path / "round_data.csv"
        valid_data.to_csv(csv_path, index=False)
        
        loaded_data = data_loader.load_csv(csv_path)
        
        # Check decimals
        price_decimals = data_loader.config["market"]["price_decimals"]
        volume_decimals = data_loader.config["market"]["volume_decimals"]
        
        for col in ['open', 'high', 'low', 'close']:
            max_decimals = loaded_data[col].astype(str).str.split('.').str[1].str.len().max()
            assert max_decimals <= price_decimals
            
        max_volume_decimals = loaded_data['volume'].astype(str).str.split('.').str[1].str.len().max()
        assert max_volume_decimals <= volume_decimals
