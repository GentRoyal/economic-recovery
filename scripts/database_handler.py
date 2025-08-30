import sqlite3
import pandas as pd
from scripts.config import settings
import os
import logging

class MyDBRepo:
    """
    A class for storing and retrieving data using SQLite.
    """
    def __init__(self):
        """
        Initializes a connection to the SQLite database.
        Sets the connection to None if an error occurs.
        """
        self.logger = logging.getLogger(__name__)
        self.db_name = settings.db_name
        absolute_path = os.path.abspath(__file__)
        directory_name = os.path.dirname(absolute_path)
        parent_name = os.path.dirname(directory_name)
        self.path = os.path.join(parent_name, 'data')
        
        try:
            self.conn = sqlite3.connect(f"{self.path}/{self.db_name}")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            self.conn = None
    
    def to_table(self, df, symbol, if_exists='replace'):
        """Saves a DataFrame to a table in the SQLite database."""
        if self.conn is None:
            self.logger.error("No database connection.")
            return
            
        try:
            df.to_sql(name=symbol, con=self.conn, if_exists=if_exists, index=True)
            self.logger.info(f"Data inserted into table '{symbol}' successfully.")
            
        except ValueError as ve:
            self.logger.error(f"Value error inserting data into table '{symbol}': {ve}")
            
        except sqlite3.Error as sql_err:
            self.logger.error(f"SQLite error inserting data into table '{symbol}': {sql_err}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error inserting data into table '{symbol}': {e}")
    
    def from_table(self, symbol):
        """Retrieves data from a specified table in the SQLite database."""
        
        if self.conn is None:
            self.logger.error("No Database Connection.")
            return pd.DataFrame()
        try:
            query = f"SELECT * FROM {symbol}"
            df = pd.read_sql(query, self.conn)
            
            df.rename(columns={'index': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df.set_index('Date', inplace=True)
            
            df.to_csv(f'{self.path}/{symbol}.csv', index=True)
            self.logger.info(f"Data retrieved from table '{symbol}' successfully.")
            
            return df
            
        except pd.io.sql.DatabaseError as db_err:
            self.logger.error(f"Database error reading from table '{symbol}': {db_err}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error reading from table '{symbol}': {e}")
            
        return pd.DataFrame()