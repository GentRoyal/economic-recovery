import pandas as pd
import logging

class RecoveryIndexCalculator:
    def __init__(self, df: pd.DataFrame, inverse_indicators = None):
        """
        df: DataFrame with indicator columns.
        inverse_indicators: List of column names where higher values are worse (to flip).
        """
        self.df = df.copy()
        self.inverse_indicators = inverse_indicators or []

        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(logging.INFO)

    def normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalize a pandas Series to 0-1 range."""
        
        try:
            min_val = series.min()
            max_val = series.max()
            
            if max_val == min_val:
                self.logger.warning(f"Max and min are equal for series '{series.name}'. Returning 0.5 for all values.")
                return series.apply(lambda x: 0.5)  # Avoid division by zero
            
            normalized = (series - min_val) / (max_val - min_val)
            self.logger.debug(f"Normalized series '{series.name}'.")
            
            return normalized
        
        except Exception as e:
            self.logger.error(f"Error normalizing series '{series.name}': {e}")
            raise

    def compute(self) -> pd.DataFrame:
        """Compute the Recovery Index and return DataFrame with a new column."""
        
        try:
            norm_df = pd.DataFrame()
            for col in self.df.columns:
                self.logger.info(f"Normalizing column '{col}'.")
                norm_col = self.normalize(self.df[col])
                if col in self.inverse_indicators:
                    self.logger.info(f"Inverting normalized values for '{col}'.")
                    norm_col = 1 - norm_col
                norm_df[col] = norm_col

            # Average normalized indicators
            norm_df['recovery_index'] = norm_df.mean(axis = 1)
            self.logger.info("Computed recovery_index.")

            # Add recovery_index to original df
            result_df = self.df.copy()
            result_df['recovery_index'] = norm_df['recovery_index']

            return result_df
        
        except Exception as e:
            self.logger.error(f"Error computing Recovery Index: {e}")
            raise