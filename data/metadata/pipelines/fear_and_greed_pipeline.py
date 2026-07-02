import pandas as pd
import numpy as np

def process_fear_greed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw Fear & Greed Index DataFrame to be compliant with fear_greed_schema.
    """
    df = df.copy()
    
    # Convert timestamp to datetime UTC
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        
    df['value'] = df['value'].astype(np.int64)
    
    # Ensure classification is string/object (categorical is also fine, but standard is object in raw data typically)
    df['classification'] = df['classification'].astype(str)
    
    # Keep only required columns
    required_cols = ['timestamp', 'value', 'classification']
    df = df[required_cols]
    
    return df
