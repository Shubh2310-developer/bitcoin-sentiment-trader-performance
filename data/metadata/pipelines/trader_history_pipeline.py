import pandas as pd
import numpy as np

def process_trader_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw trader history DataFrame to be compliant with trader_history_schema.
    """
    df = df.copy()
    
    # Filter only valid directions
    valid_directions = ['Open Long', 'Close Long', 'Open Short', 'Close Short']
    df = df[df['Direction'].isin(valid_directions)].copy()
    
    # Map Side and Direction
    df['Side'] = df['Direction'].apply(lambda x: x.split(' ')[1])
    df['Direction'] = df['Direction'].apply(lambda x: x.split(' ')[0])
    
    # Format Timestamp
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'].astype(np.float64), unit='ms', utc=True)
        
    # Ensure all required columns are present and correctly typed
    df['Trade ID'] = df['Trade ID'].astype(str)
    df['Account'] = df['Account'].astype(str)
    
    float_cols = ['Size USD', 'Execution Price', 'Closed PnL', 'Fee']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
            
    if 'Leverage' in df.columns:
        df['Leverage'] = pd.to_numeric(df['Leverage'], errors='coerce').astype(np.float64)
        
    return df
