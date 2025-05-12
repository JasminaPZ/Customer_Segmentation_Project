import numpy as np

def create_features(df):
    """Generate key customer-level features for clustering."""
    features_df = df.groupby("CustomerID").agg(
        TotalSpent=("TotalSpent", "sum"),
        Frequency=("InvoiceNo", "nunique"),
        Recency=("InvoiceDate", lambda x: (df["InvoiceDate"].max() - x.max()).days),
        AvgUnitPrice=("UnitPrice", "mean")
    ).reset_index()
    return features_df

def log_transform_features(df):
    """Apply log transformation to reduce skewness."""
    df["TotalSpent"] = np.log1p(df["TotalSpent"])
    df["Frequency"] = np.log1p(df["Frequency"])
    df["Recency"] = np.log1p(df["Recency"])
    df["AvgUnitPrice"] = np.log1p(df["AvgUnitPrice"])
    return df
