import logging
from typing import Optional, Dict, Any
from fastapi import Query
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import uvicorn
import pandas as pd
from scripts.api_data_fetcher import FREDFetcher
from scripts.database_handler import MyDBRepo
from scripts.recovery import EconomicRecoveryIndexCalculator
from datetime import date

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Economic Recovery Dashboard',
    description='API for retrieving economic indicators and recovery metrics',
    version='1.0.0'
)

app.mount("/static", StaticFiles(directory = "static"), name = "static")

@app.get("/dashboard")
async def serve_dashboard():
    """Serve the dashboard HTML file."""
    return FileResponse("static/dashboard.html")

@app.get("/")
async def root():
    """Redirect to dashboard."""
    # return {'message' : 'App is Up and Running'}
    return RedirectResponse(url = "/dashboard")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SERIES_MAP = {
    'gdp_real': 'GDPC1',
    'exports': 'EXPGS',
    'trade_balance': 'BOPGSTB',
    'jobs': 'JTSJOL',
    'unemployment': 'UNRATE',
    'total_retail_sales': 'RSXFS',
    'industrial_production' : 'IPG2211A2N'
}

INVERSE_INDICATORS = ['unemployment']


def fetch_economic_data() -> Optional[pd.DataFrame]:
    """Fetch economic data from FRED API."""
    try:
        logger.info("Fetching economic data from FRED API")
        fred = FREDFetcher()
        economic_df = fred.get_series(SERIES_MAP)
        
        if economic_df.empty:
            logger.warning("No data retrieved from FRED API")
            return None
            
        logger.info(f"Successfully fetched {len(economic_df)} records from FRED")
        return economic_df
        
    except Exception as e:
        logger.error(f"Failed to fetch data from FRED API: {e}")
        return None


def get_database_data() -> Optional[pd.DataFrame]:
    """Retrieve economic data from database."""
    try:
        logger.info("Retrieving economic data from database")
        mydb = MyDBRepo()
        economic_df = mydb.from_table('economic_indices')
        
        if economic_df.empty:
            logger.warning("No data found in database")
            return None
            
        logger.info(f"Successfully retrieved {len(economic_df)} records from database")
        return economic_df
        
    except Exception as e:
        logger.error(f"Failed to retrieve data from database: {e}")
        return None


def save_to_database(df: pd.DataFrame) -> bool:
    """Save economic data to database."""
    try:
        logger.info("Saving economic data to database")
        mydb = MyDBRepo()
        mydb.to_table(df, 'economic_indices')
        logger.info("Successfully saved data to database")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save data to database: {e}")
        return False


def compute_recovery_index(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute recovery index from economic data."""
    try:
        logger.info("Computing recovery index")
        #calculator = RecoveryIndexCalculator(df, inverse_indicators = INVERSE_INDICATORS)

        calculator = EconomicRecoveryIndexCalculator(
        reverse_indicators = ['unemployment'],
        baseline_range = (date(2019, 10, 1), date(2020, 1, 1)),
        trough_range = (date(2020, 3, 1), date(2020, 5, 31))
        )

        result_df = calculator.fit_transform(df)
        
        #result_df = calculator.compute()
        logger.info("Successfully computed recovery index")
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to compute recovery index: {e}")
        return None


def prepare_latest_data(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Prepare latest row data for API response."""
    try:
        if df.empty:
            logger.warning("DataFrame is empty, cannot prepare latest data")
            return None
            
        latest_row = df.iloc[-1].to_dict()
        latest_row['Date'] = df.index[-1].strftime('%Y-%m-%d')
        logger.info(f"Prepared latest data for date: {latest_row['Date']}")
        return latest_row
        
    except Exception as e:
        logger.error(f"Failed to prepare latest data: {e}")
        return None



@app.on_event("startup")
async def load_indicators():
    """Load economic indicators on application startup."""
    logger.info("Starting application and loading economic indicators")
    
    try:
        # Initialize app state
        app.state.latest_row = None
        app.state.full_data = None
        
        # Try to fetch fresh data from FRED
        economic_df = fetch_economic_data()
        
        if economic_df is not None:
            # Save fresh data to database
            save_success = save_to_database(economic_df)
            if not save_success:
                logger.warning("Failed to save fresh data to database, continuing with fetched data")
            
            # Retrieve data from database to ensure consistency
            economic_df = get_database_data()
            
        else:
            # Fallback to database data
            logger.info("Falling back to database data")
            economic_df = get_database_data()
        
        if economic_df is None or economic_df.empty:
            logger.error("No economic data available from either FRED or database")
            return
        
        # Compute recovery index
        result_df = compute_recovery_index(economic_df)
        if result_df is None:
            logger.error("Failed to compute recovery index")
            return
        
        # Prepare data for API responses
        latest_row = prepare_latest_data(result_df)
        if latest_row is None:
            logger.error("Failed to prepare latest data")
            return
        
        # Store in app state
        app.state.latest_row = latest_row
        app.state.full_data = result_df.reset_index()
        
        logger.info("Successfully loaded economic indicators on startup")
        
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        # Ensure app state is properly initialized even on failure
        app.state.latest_row = None
        app.state.full_data = None


@app.get("/indicators")
async def get_indicators():
    """Get the latest economic indicators."""
    try:
        if not hasattr(app.state, 'latest_row') or app.state.latest_row is None:
            logger.warning("No latest indicators data available")
            raise HTTPException(
                status_code=503, 
                detail="Economic indicators data is not available. Please try again later."
            )
        
        logger.info("Returning latest economic indicators")
        return JSONResponse(content=app.state.latest_row)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving indicators: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while retrieving indicators"
        )


@app.get("/history")
async def get_full_history():
    """Get the complete history of economic indicators."""
    try:
        if not hasattr(app.state, 'full_data') or app.state.full_data is None:
            logger.warning("No historical data available")
            raise HTTPException(
                status_code=503, 
                detail="Historical data is not available. Please try again later."
            )
        
        # Prepare data for JSON serialization
        df = app.state.full_data.copy()
        
        # Convert datetime and object columns to string
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        records = df.to_dict(orient="records")
        logger.info(f"Returning {len(records)} historical records")
        return JSONResponse(content=records)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while retrieving historical data"
        )


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "indicators_loaded": hasattr(app.state, 'latest_row') and app.state.latest_row is not None,
            "history_loaded": hasattr(app.state, 'full_data') and app.state.full_data is not None,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        if health_status["indicators_loaded"]:
            health_status["latest_date"] = app.state.latest_row.get('Date', 'Unknown')
            
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500
        )

@app.get("/indicators-range")
async def get_indicators_range(
    start_date: str = Query(..., example="2020-01-01"),
    end_date: str = Query(..., example="2024-12-31")
):
    """
    Fetch economic indicators from FRED based on a user-selected date range.
    This does NOT modify the cached app.state data.
    """
    try:
        logger.info(f"Fetching FRED data from {start_date} to {end_date}")
        
        fred = FREDFetcher(start_date = start_date, end_date = end_date)
        economic_df = fred.get_series(SERIES_MAP, observation_start = start_date, observation_end = end_date)

        if economic_df.empty:
            logger.warning("No data retrieved for selected range")
            raise HTTPException(status_code=404, detail="No data found for the given date range")

        result_df = compute_recovery_index(economic_df)
        if result_df is None or result_df.empty:
            logger.error("Failed to compute recovery index for selected range")
            raise HTTPException(status_code=500, detail="Failed to compute recovery index")

        latest_row = prepare_latest_data(result_df)
        if latest_row is None:
            logger.error("Failed to prepare latest row for selected range")
            raise HTTPException(status_code=500, detail="Failed to prepare latest data")

        # Convert result_df to records for frontend
        result_df = result_df.reset_index()

        # Rename 'index' to 'Date'
        if 'index' in result_df.columns:
            result_df = result_df.rename(columns={'index': 'Date'})
        
        # Format 'Date' properly
        result_df['Date'] = pd.to_datetime(result_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Drop any rows where 'Date' couldn't be parsed
        result_df = result_df.dropna(subset=['Date'])
        
        # Convert to records
        history_records = result_df.to_dict(orient="records")


        logger.info(f"Returning {len(history_records)} records for selected range")

        return {
            "latest": latest_row,
            "history": history_records
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /indicators-range: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)