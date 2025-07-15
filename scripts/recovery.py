import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import date
from typing import List, Tuple, Optional


class EconomicRecoveryIndexCalculator:
    """
    A class to calculate economic recovery index using PCA-weighted recovery ratios.
    
    The recovery index measures how well economic indicators have recovered from
    a trough period relative to a baseline period, using PCA weights to combine
    multiple indicators into a single index.
    """
    
    def __init__(self, 
                 indicators: Optional[List[str]] = None,
                 reverse_indicators: Optional[List[str]] = None,
                 baseline_range: Tuple[date, date] = (date(2019, 10, 1), date(2020, 1, 1)),
                 trough_range: Tuple[date, date] = (date(2020, 3, 1), date(2020, 5, 31))):
        """
        Initialize the calculator with configuration parameters.
        
        Args:
            indicators: List of economic indicator column names. If None, will be inferred from data.
            reverse_indicators: List of indicators where lower values are better (e.g., unemployment).
            baseline_range: Tuple of (start_date, end_date) for baseline period.
            trough_range: Tuple of (start_date, end_date) for trough period.
        """
        self.indicators = indicators
        self.reverse_indicators = reverse_indicators or ['unemployment']
        self.baseline_range = baseline_range
        self.trough_range = trough_range
        
        # These will be set during fitting
        self.baseline = None
        self.trough = None
        self.pca_weights = None
        self.scaler = None
        self.pca = None
        self.min_val = None
        self.max_val = None
        self.is_fitted = False
    
    def fit(self, economic_df: pd.DataFrame) -> 'EconomicRecoveryIndexCalculator':
        """
        Fit the calculator on the provided economic data.
        
        Args:
            economic_df: DataFrame with economic indicators and date index.
            
        Returns:
            self: Returns the fitted calculator instance.
        """
        # Set indicators if not provided
        if self.indicators is None:
            self.indicators = economic_df.columns.tolist()
        
        # Calculate baseline and trough averages
        self.baseline = economic_df.loc[self.baseline_range[0]:self.baseline_range[1]][self.indicators].mean()
        self.trough = economic_df.loc[self.trough_range[0]:self.trough_range[1]][self.indicators].mean()
        
        # Fit PCA to get weights
        self._fit_pca(economic_df)
        
        # Calculate recovery ratios for scaling bounds
        recovery_ratios_df = economic_df[self.indicators].apply(self._calculate_recovery_ratio, axis=1)
        recovery_index = recovery_ratios_df.dot(self.pca_weights)
        
        # Store scaling parameters
        self.min_val = recovery_index.min()
        self.max_val = recovery_index.max()
        
        self.is_fitted = True
        return self
    
    def transform(self, economic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the economic data by adding the recovery index.
        
        Args:
            economic_df: DataFrame with economic indicators and date index.
            
        Returns:
            DataFrame with added 'recovery_index' column (scaled 0-100).
        """
        if not self.is_fitted:
            raise ValueError("Calculator must be fitted before transform. Call fit() first.")
        
        # Calculate recovery ratios
        recovery_ratios_df = economic_df[self.indicators].apply(self._calculate_recovery_ratio, axis=1)
        
        # Calculate weighted recovery index
        recovery_index = recovery_ratios_df.dot(self.pca_weights)
        
        # Scale to 0-100 range
        scaled_index = 100 * (recovery_index - self.min_val) / (self.max_val - self.min_val)
        
        # Add to DataFrame
        result_df = economic_df.copy()
        result_df['recovery_index'] = scaled_index
        
        return result_df
    
    def fit_transform(self, economic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the calculator and transform the data in one step.
        
        Args:
            economic_df: DataFrame with economic indicators and date index.
            
        Returns:
            DataFrame with added 'recovery_index' column (scaled 0-100).
        """
        return self.fit(economic_df).transform(economic_df)
    
    def _fit_pca(self, economic_df: pd.DataFrame) -> None:
        """Fit PCA to extract weights for combining indicators."""
        # Prepare data for PCA
        X = economic_df[self.indicators].dropna()
        
        # Standardize data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA and extract first principal component weights
        self.pca = PCA(n_components=1)
        self.pca.fit(X_scaled)
        
        # Create normalized weights
        self.pca_weights = pd.Series(
            self.pca.components_[0],
            index=self.indicators,
            name='pca_weight'
        )
        self.pca_weights = self.pca_weights / self.pca_weights.sum()
    
    def _calculate_recovery_ratio(self, row: pd.Series) -> pd.Series:
        """
        Calculate recovery ratio for a single row of data.
        
        Args:
            row: Series containing values for all indicators.
            
        Returns:
            Series with recovery ratios for each indicator.
        """
        ratios = {}
        
        for col in self.indicators:
            b = self.baseline[col]  # baseline value
            t = self.trough[col]    # trough value
            c = row[col]            # current value
            
            if col in self.reverse_indicators:
                # For reverse indicators (lower is better)
                ratio = (t - c) / (t - b) if (t - b) != 0 else 1
            else:
                # For normal indicators (higher is better)
                ratio = (c - t) / (b - t) if (b - t) != 0 else 1
            
            # Clip at 0 (no negative recovery)
            ratios[col] = max(0, ratio)
        
        return pd.Series(ratios)
    
    def get_pca_weights(self) -> pd.Series:
        """
        Get the PCA weights used for combining indicators.
        
        Returns:
            Series with PCA weights for each indicator.
        """
        if not self.is_fitted:
            raise ValueError("Calculator must be fitted first.")
        return self.pca_weights.copy()
    
    def get_baseline_trough_values(self) -> Tuple[pd.Series, pd.Series]:
        """
        Get the baseline and trough values used for calculations.
        
        Returns:
            Tuple of (baseline_values, trough_values) as Series.
        """
        if not self.is_fitted:
            raise ValueError("Calculator must be fitted first.")
        return self.baseline.copy(), self.trough.copy()