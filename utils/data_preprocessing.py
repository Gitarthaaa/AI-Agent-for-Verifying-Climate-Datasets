import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.valid_ranges: Dict[str, Dict[str, float]] = {
            'temperature': {'min': -90, 'max': 60},  # °C
            'humidity': {'min': 0, 'max': 100},      # %
            'precipitation': {'min': 0, 'max': 2000}, # mm
            'pressure': {'min': 870, 'max': 1090},   # hPa
            'wind_speed': {'min': 0, 'max': 408}     # km/h (world record is 407 km/h)
        }

    def validate_data_structure(self, data: pd.DataFrame) -> List[str]:
        """Check if the data has required columns and proper structure."""
        issues = []
        required_columns = ['Timestamp', 'Location']
        
        for col in required_columns:
            if col not in data.columns:
                issues.append(f"Missing required column: {col}")
        
        if 'timestamp' in data.columns:
            try:
                pd.to_datetime(data['timestamp'])
            except:
                issues.append("Invalid timestamp format")
        
        return issues

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the dataset."""
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            data[numeric_cols] = self.numeric_imputer.fit_transform(data[numeric_cols])
        if len(categorical_cols) > 0:
            data[categorical_cols] = self.categorical_imputer.fit_transform(data[categorical_cols])
        
        return data

    def validate_ranges(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """Check if values are within expected ranges."""
        anomalies = {}
        
        for column, ranges in self.valid_ranges.items():
            if column in data.columns:
                invalid_indices = data[
                    (data[column] < ranges['min']) | 
                    (data[column] > ranges['max'])
                ].index.tolist()
                if invalid_indices:
                    anomalies[column] = invalid_indices
        
        return anomalies

    def check_temporal_consistency(self, data: pd.DataFrame) -> List[int]:
        """Check for temporal consistency in time series data."""
        if 'timestamp' not in data.columns:
            return []
        
        data = data.sort_values('timestamp')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['time_diff'] = data['timestamp'].diff()
        
        # Flag records with unexpected time gaps (e.g., more than 24 hours)
        inconsistent_indices = data[data['time_diff'] > pd.Timedelta(hours=24)].index.tolist()
        
        return inconsistent_indices

    def detect_duplicates(self, data: pd.DataFrame) -> List[int]:
        """Identify duplicate records."""
        return data[data.duplicated()].index.tolist()

    def process_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
        """Main method to run all preprocessing and validation checks."""
        validation_report = {
            'structure_issues': self.validate_data_structure(data),
            'range_anomalies': {},
            'temporal_inconsistencies': [],
            'duplicates': []
        }
        
        if not validation_report['structure_issues']:
            # Only proceed with other checks if structure is valid
            data = self.handle_missing_values(data)
            validation_report['range_anomalies'] = self.validate_ranges(data)
            validation_report['temporal_inconsistencies'] = self.check_temporal_consistency(data)
            validation_report['duplicates'] = self.detect_duplicates(data)
        
        return data, validation_report