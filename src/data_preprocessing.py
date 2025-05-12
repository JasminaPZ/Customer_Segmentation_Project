import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the raw dataset from the specified path."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the raw data by removing duplicates, handling missing values, and filtering UK customers."""
    df = df.dropna(subset=["CustomerID"])
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df["Country"] == "United Kingdom"].reset_index(drop=True)
    return df

def remove_canceled_invoices(df):
    """Remove canceled invoices based on InvoiceNo prefix 'C'."""
    return df[~df["InvoiceNo"].str.startswith("C")].reset_index(drop=True)

def save_processed_data(df, file_path):
    """Save the processed data to the specified path."""
    df.to_csv(file_path, index=False)
