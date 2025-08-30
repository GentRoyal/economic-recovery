import logging
import json
import pandas as pd
import requests
from fredapi import Fred
from typing import List, Dict, Optional

from scripts.config import settings

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("api_data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FREDFetcher:
    """Fetches economic data series from the FRED API."""

    def __init__(self, start_date: str = "2019-01-01", end_date: Optional[str] = None):
        self.fred = Fred(api_key = settings.api_key)
        self.start_date = start_date
        self.end_date = end_date

    def get_series(self, series_map: Dict[str, str], observation_start: Optional[str] = None, observation_end: Optional[str] = None) -> pd.DataFrame:
        """Fetch multiple series and return as a DataFrame."""
        start = observation_start or self.start_date
        end = observation_end or self.end_date

        data = {}
        for name, series_id in series_map.items():
            logger.info(f"Fetching FRED series: {series_id} as '{name}'")
            
            try:
                series_data = self.fred.get_series(series_id, observation_start = start, observation_end = end)
                data[name] = series_data
                logger.debug(f"Retrieved {len(series_data)} records for {name}")
                
            except Exception as e:
                logger.error(f"Failed to fetch FRED series '{series_id}': {e}")
                data[name] = pd.Series(dtype='float64')  # fallback empty series

        df = pd.DataFrame(data)
        df.sort_index(inplace = True)
        df.fillna(method = 'ffill', inplace = True)
        
        logger.info(f"FRED fetch complete. Columns: {list(df.columns)}")
        return df
