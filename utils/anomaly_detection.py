import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Union, Optional

class AnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def detect_statistical_anomalies(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect anomalies using statistical methods (Z-score)."""
        anomalies = {}
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            anomalies[col] = data[z_scores > 3].index.tolist()  # 3 standard deviations
            
        return anomalies
    
    def detect_isolation_forest_anomalies(self, data: pd.DataFrame) -> List[int]:
        """Detect anomalies using Isolation Forest."""
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            return []
            
        try:
            scaled_data = self.scaler.fit_transform(numeric_data)
            predictions = self.isolation_forest.fit_predict(scaled_data)
            return data[predictions == -1].index.tolist()
        except Exception as e:
            print(f"Error in isolation forest detection: {e}")
            return []
    
    def detect_temporal_anomalies(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect sudden changes or spikes in time series data."""
        anomalies = {}
        if 'timestamp' not in data.columns:
            return anomalies
            
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data = data.sort_values('timestamp')
        
        for col in numeric_cols:
            if col == 'timestamp':
                continue
                
            # Calculate rolling statistics
            rolling_mean = data[col].rolling(window=5).mean()
            rolling_std = data[col].rolling(window=5).std()
            
            # Detect points that deviate significantly from rolling statistics
            deviations = np.abs(data[col] - rolling_mean) / rolling_std
            anomalies[col] = data[deviations > 3].index.tolist()
            
        return anomalies
    
    def detect_correlation_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Union[int, str]]]:
        """Detect anomalies in correlations between variables."""
        anomalies = []
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) < 2:
            return anomalies
            
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Find pairs with unexpected correlations
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                correlation = corr_matrix.iloc[i, j]
                
                # Flag very strong correlations (might indicate data quality issues)
                if abs(correlation) > 0.95:
                    anomalies.append({
                        'type': 'high_correlation',
                        'columns': [col1, col2],
                        'correlation': correlation
                    })
                    
        return anomalies

def detect_anomalies(data: pd.DataFrame) -> Dict[str, Union[List, Dict]]:
    """Main function to detect all types of anomalies in the dataset."""
    detector = AnomalyDetector()
    
    anomaly_report = {
        'statistical_anomalies': {},
        'isolation_forest_anomalies': [],
        'temporal_anomalies': {},
        'correlation_anomalies': [],
    }
    
    if isinstance(data, pd.DataFrame):
        anomaly_report['statistical_anomalies'] = detector.detect_statistical_anomalies(data)
        anomaly_report['isolation_forest_anomalies'] = detector.detect_isolation_forest_anomalies(data)
        anomaly_report['temporal_anomalies'] = detector.detect_temporal_anomalies(data)
        anomaly_report['correlation_anomalies'] = detector.detect_correlation_anomalies(data)
    
    return anomaly_report
